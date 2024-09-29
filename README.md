```
python ws_server.py --base_model_path "digiplay/GhostMix" --acceleration "tensorrt" --prompt "" --num_inference_steps 30 --guidance_scale 0.6 --lora_path ../models/RetroGlitch.safetensors --lora_scale 1.2 --t_index_list "[8,20]"
```

```
python ws_client.py --prompt "masterpiece, best quality, solo, ((Red Retro Glitch)), dark, techno, Retro Glitch, a man with a head made of many different colored balls "
```

For pi the easiest is to install picamera and opencv without venvs.