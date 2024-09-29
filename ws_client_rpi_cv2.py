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
        print("Starting picamera2...")
        picam2.start()

        # Setup cv2 window
        cv2.namedWindow("image", cv2.WINDOW_NORMAL)
        if fullscreen:
            cv2.setWindowProperty("image", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

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
            frame = cv2.resize(frame, (image_size, image_size))

            # Encode frame
            _, buffer = cv2.imencode(".jpg", frame)
            jpg_as_text = base64.b64encode(buffer).decode("utf-8")

            # Send to server
            await websocket.send(jpg_as_text)

            # Receive and display image
            response = await websocket.recv()
            img = base64.b64decode(response)
            npimg = np.frombuffer(img, dtype=np.uint8)
            source = cv2.imdecode(npimg, 1)

            # Flip
            source = cv2.flip(source, 1)

            # Resize
            source = cv2.resize(
                source, dsize=(1400, 1400), interpolation=cv2.INTER_CUBIC
            )

            cv2.imshow("image", source)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            # Wait
            await asyncio.sleep(0.0001)

        cv2.destroyAllWindows()


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
            args.fullscreen,
            args.crop_size,
            args.crop_offset_y,
            args.compression,
        )
    )
