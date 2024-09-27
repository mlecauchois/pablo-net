import asyncio
import websockets
import base64
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
import json
import io
import cv2
from picamera2 import Picamera2


async def capture_and_send(
    url,
    prompt,
    negative_prompt,
    image_size=256,
    rotate=0,
    fullscreen=False,
    crop_size=256,
    crop_offset_y=0,
    compression=90,
):
    uri = url
    async with websockets.connect(uri) as websocket:
        # Initialize picamera2
        picam2 = Picamera2()
        # Full resolution preview
        picam2.configure(
            picam2.create_preview_configuration(main={"size": (1400, 1000)})
        )
        picam2.start()

        # Setup Tkinter window
        root = tk.Tk()
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        if fullscreen:
            root.attributes("-fullscreen", True)
        else:
            root.geometry(f"{screen_width}x{screen_height}")
        label = tk.Label(root)
        label.pack(expand=True, fill="both")

        print("Connected to server...")

        # Send prompt to server as json
        await websocket.send(
            json.dumps(
                {
                    "prompt": prompt,
                    "negative_prompt": negative_prompt,
                }
            )
        )

        async def update_frame():
            while True:
                # Capture frame
                frame = picam2.capture_array()
                h, w, _ = frame.shape

                frame = Image.fromarray(frame)
                frame = frame.convert("RGB")
                frame = np.array(frame)

                # Crop square of crop_size in the middle with offset
                frame = frame[
                    h // 2
                    - crop_size // 2
                    - crop_offset_y : h // 2
                    + crop_size // 2
                    - crop_offset_y,
                    w // 2 - crop_size // 2 : w // 2 + crop_size // 2,
                ]

                # Reduce size
                frame = Image.fromarray(frame).resize((image_size, image_size))
                frame = np.array(frame)

                # Encode frame as JPEG using PIL
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), compression]
                result, buffer = cv2.imencode(".jpg", frame, encode_param)
                jpg_as_text = base64.b64encode(buffer).decode("utf-8")

                # Send to server
                await websocket.send(jpg_as_text)

                # Receive and display image
                response = await websocket.recv()
                img_data = base64.b64decode(response)
                source = Image.open(io.BytesIO(img_data))

                # Flip image
                source = source.transpose(Image.FLIP_LEFT_RIGHT)

                # Resize image to screen dimensions
                # First crop the image to the correct aspect ratio
                source_aspect = source.width / source.height
                target_aspect = screen_width / screen_height
                if source_aspect > target_aspect:
                    # Crop the width
                    new_width = int(target_aspect * source.height)
                    offset = (source.width - new_width) // 2
                    source = source.crop((offset, 0, offset + new_width, source.height))
                else:
                    # Crop the height
                    new_height = int(source.width / target_aspect)
                    offset = (source.height - new_height) // 2
                    source = source.crop((0, offset, source.width, offset + new_height))
                source = source.resize((screen_width, screen_height), Image.BICUBIC)

                # Rotate if angle is provided
                if rotate != 0:
                    source = source.rotate(rotate)

                # Update Tkinter label with the new image
                tk_image = ImageTk.PhotoImage(source)
                label.config(image=tk_image)
                label.image = tk_image  # Keep a reference

                # Update the GUI
                root.update()

                # Wait
                await asyncio.sleep(0.0001)

        # Run the update_frame coroutine
        await update_frame()

        picam2.stop()
        root.destroy()


# Command args cli
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--url", type=str, help="URL of server", default="")
    parser.add_argument("--prompt", type=str, help="Prompt to send to server")
    parser.add_argument(
        "--negative_prompt",
        type=str,
        help="Negative prompt to send to server",
        default="low quality",
    )
    parser.add_argument("--image_size", type=int, help="Image size", default=256)
    parser.add_argument(
        "--rotate", type=float, default=0, help="Rotate the image by specified degrees"
    )
    parser.add_argument(
        "--fullscreen", action="store_true", help="Display window in fullscreen mode"
    )
    parser.add_argument(
        "--crop_size", type=int, default=256, help="Crop size of the image"
    )
    parser.add_argument(
        "--crop_offset_y",
        type=int,
        default=0,
        help="Offset of the crop from the top of the image",
    )
    parser.add_argument(
        "--compression", type=int, default=90, help="JPEG compression quality"
    )

    args = parser.parse_args()

    asyncio.get_event_loop().run_until_complete(
        capture_and_send(
            args.url,
            args.prompt,
            args.negative_prompt,
            args.image_size,
            args.rotate,
            args.fullscreen,
            args.crop_size,
            args.crop_offset_y,
            args.compression,
        )
    )
