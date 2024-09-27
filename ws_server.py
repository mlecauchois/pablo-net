import fire
import cv2
import numpy as np
import time
import asyncio
import websockets
import cv2
import numpy as np
import base64
import json

import torch
from diffusers import AutoencoderTiny, StableDiffusionPipeline

from streamdiffusion import StreamDiffusion
from streamdiffusion.image_utils import postprocess_image
from streamdiffusion.acceleration.tensorrt import accelerate_with_tensorrt


import PIL.Image


# Initialize and load the model outside the process_image function
def load_model(
    base_model_path, acceleration, lora_path=None, lora_scale=1.0, t_index_list=None
):
    pipe = StableDiffusionPipeline.from_pretrained(base_model_path).to(
        device=torch.device("cuda"),
        dtype=torch.float16,
    )

    stream = StreamDiffusion(
        pipe,
        t_index_list=t_index_list,
        torch_dtype=torch.float16,
        width=512,
        height=512,
    )

    stream.enable_similar_image_filter(
        0.98,
        10,
    )

    stream.load_lcm_lora()
    stream.fuse_lora()

    if lora_path is not None:
        stream.load_lora(lora_path)
        stream.fuse_lora(lora_scale=lora_scale)
        print(f"Using LoRA: {lora_path}")

    stream.vae = AutoencoderTiny.from_pretrained("madebyollin/taesd").to(
        device=pipe.device, dtype=pipe.dtype
    )

    if acceleration == "tensorrt":
        stream = accelerate_with_tensorrt(
            stream,
            "engines",
            max_batch_size=2,
        )
    else:
        pipe.enable_xformers_memory_efficient_attention()

    return stream


async def process_image(
    websocket,
    stream,
    prompt,
    num_inference_steps=50,
    preprocessing=None,
    negative_prompt="",
    guidance_scale=1.2,
    compression=30,
):
    # Prepare the stream
    stream.prepare(
        prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        negative_prompt=negative_prompt,
    )

    async for message in websocket:
        try:
            message = json.loads(message)
            new_prompt = message["prompt"]
            new_negative_prompt = message["negative_prompt"]
            stream.prepare(
                prompt=new_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                negative_prompt=new_negative_prompt,
            )
            continue
        except:
            pass

        # Decode the image
        nparr = np.frombuffer(base64.b64decode(message), np.uint8)
        image_size_in_bytes = nparr.nbytes
        print(f"Image size: {image_size_in_bytes} bytes")

        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if preprocessing == "canny":
            img = cv2.Canny(img, 100, 200)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif preprocessing == "canny_blur_shift":
            blur_img = cv2.GaussianBlur(img, (3, 3), 0)
            canny_img = cv2.Canny(blur_img, 100, 200)
            canny_img = cv2.cvtColor(canny_img, cv2.COLOR_GRAY2BGR)
            # Change black to blue
            canny_img[np.where((canny_img == [0, 0, 0]).all(axis=2))] = [255, 0, 0]
            # Add
            img = cv2.addWeighted(img, 0.8, canny_img, 0.2, 0)
        elif preprocessing == "blur":
            img = cv2.GaussianBlur(img, (5, 5), 0)
        elif preprocessing == "gray":
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif preprocessing == "contrast":
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            img = clahe.apply(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        # Convert to PIL
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = PIL.Image.fromarray(img)

        start = time.time()
        x_output = stream(img)
        output = postprocess_image(x_output, output_type="pil")[0]
        print(f"Time: {time.time() - start}")

        # Convert pil to numpy array
        output = np.array(output)
        output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)

        # Encode and send back
        compression_params = [int(cv2.IMWRITE_JPEG_QUALITY), compression]
        _, buffer = cv2.imencode(".jpg", output, compression_params)
        jpg_as_text = base64.b64encode(buffer).decode("utf-8")

        # Calculate the size of the image
        output_size_in_bytes = len(jpg_as_text)
        print(f"Output size: {output_size_in_bytes} bytes")
        await websocket.send(jpg_as_text)


def run_server(
    base_model_path,
    acceleration,
    prompt,
    host="0.0.0.0",
    port=5678,
    num_inference_steps=50,
    preprocessing=None,
    negative_prompt="",
    guidance_scale=1.2,
    lora_path=None,
    lora_scale=1.0,
    t_index_list=None,
    compression=30,
):
    # Load the model before starting the server
    stream = load_model(
        base_model_path, acceleration, lora_path, lora_scale, t_index_list
    )

    start_server = websockets.serve(
        lambda ws: process_image(
            ws,
            stream,
            prompt,
            num_inference_steps,
            preprocessing,
            negative_prompt=negative_prompt,
            guidance_scale=guidance_scale,
            compression=compression,
        ),
        host,
        port,
    )

    asyncio.get_event_loop().run_until_complete(start_server)
    asyncio.get_event_loop().run_forever()


if __name__ == "__main__":
    fire.Fire(run_server)
