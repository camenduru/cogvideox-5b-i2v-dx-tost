import os, json, requests, random, time, runpod
from urllib.parse import urlsplit

import torch
from PIL import Image
import numpy as np

import asyncio
import execution
import server
loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)
server_instance = server.PromptServer(loop)
execution.PromptQueue(server)

from nodes import load_custom_node
from nodes import NODE_CLASS_MAPPINGS

load_custom_node("/content/ComfyUI/custom_nodes/ComfyUI-CogVideoXWrapper")
load_custom_node("/content/ComfyUI/custom_nodes/ComfyUI-VideoHelperSuite")
load_custom_node("/content/ComfyUI/custom_nodes/ComfyUI-KJNodes")

LoadImage = NODE_CLASS_MAPPINGS["LoadImage"]()
ImageResizeKJ = NODE_CLASS_MAPPINGS["ImageResizeKJ"]()
CogVideoImageEncode = NODE_CLASS_MAPPINGS["CogVideoImageEncode"]()
CogVideoLoraSelect = NODE_CLASS_MAPPINGS["CogVideoLoraSelect"]()
DownloadAndLoadCogVideoModel = NODE_CLASS_MAPPINGS["DownloadAndLoadCogVideoModel"]()
CogVideoTextEncode = NODE_CLASS_MAPPINGS["CogVideoTextEncode"]()
CLIPLoader = NODE_CLASS_MAPPINGS["CLIPLoader"]()
CogVideoSampler = NODE_CLASS_MAPPINGS["CogVideoSampler"]()
CogVideoDecode = NODE_CLASS_MAPPINGS["CogVideoDecode"]()
VHS_VideoCombine = NODE_CLASS_MAPPINGS["VHS_VideoCombine"]()

with torch.inference_mode():
    lora = CogVideoLoraSelect.getlorapath("orbit_up_lora_weights.safetensors", 1.0, fuse_lora=True)[0]
    pipeline = DownloadAndLoadCogVideoModel.loadmodel("THUDM/CogVideoX-5b-I2V", "bf16", fp8_transformer="disabled", compile="disabled", enable_sequential_cpu_offload=False, lora=lora)[0]
    clip = CLIPLoader.load_clip("t5xxl_fp16.safetensors", type="sd3")[0]

def download_file(url, save_dir, file_name):
    os.makedirs(save_dir, exist_ok=True)
    original_file_name = url.split('/')[-1]
    _, original_file_extension = os.path.splitext(original_file_name)
    file_path = os.path.join(save_dir, file_name + original_file_extension)
    response = requests.get(url)
    response.raise_for_status()
    with open(file_path, 'wb') as file:
        file.write(response.content)
    return file_path

@torch.inference_mode()
def generate(input):
    values = input["input"]

    input_image=values['input_image_check']
    input_image=download_file(url=input_image, save_dir='/content/ComfyUI/input', file_name='input_image')
    prompt = values['prompt']
    negative_prompt = values['negative_prompt']
    seed = values['seed']
    steps = values['steps']
    cfg = values['cfg']

    if seed == 0:
        random.seed(int(time.time()))
        seed = random.randint(0, 18446744073709551615)

    positive = CogVideoTextEncode.process(clip, prompt, strength=1.0, force_offload=True)[0]
    negative = CogVideoTextEncode.process(clip, negative_prompt, strength=1.0, force_offload=True)[0]

    image, _ = LoadImage.load_image(input_image)
    image = ImageResizeKJ.resize(image, width=720, height=480, keep_proportion=False, upscale_method="lanczos", divisible_by=16, crop="center")[0]
    image_cond_latents = CogVideoImageEncode.encode(pipeline, image, chunk_size=16, enable_tiling=True)[0]
    samples = CogVideoSampler.process(pipeline, positive, negative, steps, cfg, seed, height=480, width=720, num_frames=49, scheduler="CogVideoXDPMScheduler", denoise_strength=1.0, image_cond_latents=image_cond_latents)
    frames = CogVideoDecode.decode(samples[0], samples[1], enable_vae_tiling=True, tile_sample_min_height=240, tile_sample_min_width=360, tile_overlap_factor_height=0.2, tile_overlap_factor_width=0.2, auto_tile_size=True)[0]

    out_video = VHS_VideoCombine.combine_video(images=frames, frame_rate=8, loop_count=0, filename_prefix="CogVideoX-I2V", format="video/h264-mp4", save_output=True)
    source = out_video["result"][0][1][1]
    destination = f"/content/ComfyUI/output/cogvideox-5b-i2v-dimensionx-{seed}-tost.mp4"
    shutil.move(source, destination)

    result = f"/content/ComfyUI/output/cogvideox-5b-i2v-dimensionx-{seed}-tost.mp4"
    try:
        notify_uri = values['notify_uri']
        del values['notify_uri']
        notify_token = values['notify_token']
        del values['notify_token']
        discord_id = values['discord_id']
        del values['discord_id']
        if(discord_id == "discord_id"):
            discord_id = os.getenv('com_camenduru_discord_id')
        discord_channel = values['discord_channel']
        del values['discord_channel']
        if(discord_channel == "discord_channel"):
            discord_channel = os.getenv('com_camenduru_discord_channel')
        discord_token = values['discord_token']
        del values['discord_token']
        if(discord_token == "discord_token"):
            discord_token = os.getenv('com_camenduru_discord_token')
        job_id = values['job_id']
        del values['job_id']
        default_filename = os.path.basename(result)
        with open(result, "rb") as file:
            files = {default_filename: file.read()}
        payload = {"content": f"{json.dumps(values)} <@{discord_id}>"}
        response = requests.post(
            f"https://discord.com/api/v9/channels/{discord_channel}/messages",
            data=payload,
            headers={"Authorization": f"Bot {discord_token}"},
            files=files
        )
        response.raise_for_status()
        result_url = response.json()['attachments'][0]['url']
        notify_payload = {"jobId": job_id, "result": result_url, "status": "DONE"}
        web_notify_uri = os.getenv('com_camenduru_web_notify_uri')
        web_notify_token = os.getenv('com_camenduru_web_notify_token')
        if(notify_uri == "notify_uri"):
            requests.post(web_notify_uri, data=json.dumps(notify_payload), headers={'Content-Type': 'application/json', "Authorization": web_notify_token})
        else:
            requests.post(web_notify_uri, data=json.dumps(notify_payload), headers={'Content-Type': 'application/json', "Authorization": web_notify_token})
            requests.post(notify_uri, data=json.dumps(notify_payload), headers={'Content-Type': 'application/json', "Authorization": notify_token})
        return {"jobId": job_id, "result": result_url, "status": "DONE"}
    except Exception as e:
        error_payload = {"jobId": job_id, "status": "FAILED"}
        try:
            if(notify_uri == "notify_uri"):
                requests.post(web_notify_uri, data=json.dumps(error_payload), headers={'Content-Type': 'application/json', "Authorization": web_notify_token})
            else:
                requests.post(web_notify_uri, data=json.dumps(error_payload), headers={'Content-Type': 'application/json', "Authorization": web_notify_token})
                requests.post(notify_uri, data=json.dumps(error_payload), headers={'Content-Type': 'application/json', "Authorization": notify_token})
        except:
            pass
        return {"jobId": job_id, "result": f"FAILED: {str(e)}", "status": "FAILED"}
    finally:
        if os.path.exists(result):
            os.remove(result)
        if os.path.exists(input_image):
            os.remove(input_image)

runpod.serverless.start({"handler": generate})