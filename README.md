# 🎓 Fundamentals of Computer Graphics – Full Projects

This repository contains **three assignments** developed by *Edoardo Papa* (Matricola 1962169) during the **Fundamentals of Computer Graphics** course. Each project demonstrates a specific set of graphics techniques using C++ and CMake, ranging from ray tracing to advanced shading and complex scene rendering.

---

## 🗂 Project Overview

```text
GITHUB/
├── Homework1/ – Ray Tracing with Toon and Refraction Effects
├── Homework2/ – Path Tracing, Refraction, and Large Scenes
├── Homework3/ – 
```

---

## 🧪 Homework 1 – Basic Raytracing with Shader Effects

### ✨ Features Implemented
- **Toon Shader (Cell Shading)** using light quantization and custom shader logic
- **Refraction Shader** simulating Fresnel effect with reflect and refract blending
- Configurable resolution and samples

### 🧠 Implementation Notes
For the **Refraction Shader**, a new `refractor` shader was added to the `raytrace_shader_type` enum. The shader computes reflection and transmission based on the Fresnel term and uses random sampling to switch between them.

For the **Toon Shader**, the logic from the `shade_raytrace` was adapted, and a `get_toon_magic()` function was created based on the algorithm from:
> 🔗 [https://roystan.net/articles/toon-shader.html](https://roystan.net/articles/toon-shader.html)

### 🖼 Example Outputs

- **Refraction Example**  
  ![Refract Effect](https://raw.githubusercontent.com/DarkShrill/Fundamentals-of-Computer-Graphics-Homework-Projects/master/Homework1/Consegna%20Homework%201/0x_refract_720_256.jpg)

- **Toon Effect Example**  
  ![Toon Effect](https://raw.githubusercontent.com/DarkShrill/Fundamentals-of-Computer-Graphics-Homework-Projects/master/Homework1/Consegna%20Homework%201/0x_toon_effect_720_256.jpg)

### ▶️ Run Examples
```bash
./bin/yraytrace --scene tests/08_glass/glass.json --output out/lowres/0x_refract_720_9.jpg --samples 256 --shader refractor --resolution 720

./bin/yraytrace --scene tests/07_plastic/plastic.json --output out/lowres/0x_ownshader_720_9.jpg --samples 256 --shader ownshader --resolution 720
```

---

## 🔬 Homework 2 – Path Tracing, Large Scenes, and MYOS

### ✨ Extra Features
- **Refraction in Path Tracing**: integrated into `eval_bsdfcos`, `eval_delta`, `sample_delta`, and others
- **Large Scene Rendering**: scenes like San Miguel, Bistro, Classroom rendered at 1280px with 1024 samples
- **Custom Scene (MYOS)**: built using Blender, JSON editing, and texture mapping

### 🖼 Example Outputs

- **Refraction Cornell Box**  
  `out/path/01_cornellbox_refractive_s512_r1024.jpg`

- **San Miguel Scene**  
  `out/path/sanmiguel_s1024_r1280.jpg`

- **Custom MYOS Scene**  
  `out/path/myos.jpg`

### ▶️ Run Examples
```bash
./bin/ypathtrace --scene tests/01_cornellbox_refractive/cornellbox.json --output out/path/01_cornellbox_refractive_s512_r1024.jpg --shader pathtrace --samples 512 --resolution 1024 --bounces 8

./bin/ypathtrace --scene tests/sanmiguel/sanmiguel.json --output out/path/sanmiguel_s1024_r1280.jpg --shader pathtrace --samples 1024 --resolution 1280 --bounces 8

./bin/ypathtrace --scene tests/MYOS/myos.json --output out/path/myos.jpg --shader pathtrace --samples 1024 --resolution 1024 --bounces 8
```

---

## 🛠 Requirements

- CMake ≥ 3.10
- C++17 compatible compiler
- OpenCV (optional, for image handling)
- Blender (for model prep)

---

## 👤 Author

**Edoardo Papa**  
Matricola: 1962169  
📘 Course: Fundamentals of Computer Graphics

---

## 📄 License

This project is intended for educational and academic use only.
