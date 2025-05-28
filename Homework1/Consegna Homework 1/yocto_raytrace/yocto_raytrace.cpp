//
// Implementation for Yocto/RayTrace.
//

//
// LICENSE:
//
// Copyright (c) 2016 -- 2021 Fabio Pellacini
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//

#include "yocto_raytrace.h"

#include <yocto/yocto_cli.h>
#include <yocto/yocto_geometry.h>
#include <yocto/yocto_parallel.h>
#include <yocto/yocto_sampling.h>
#include <yocto/yocto_shading.h>
#include <yocto/yocto_shape.h>

#include <windows.h>

// -----------------------------------------------------------------------------
// IMPLEMENTATION FOR SCENE EVALUATION
// -----------------------------------------------------------------------------
namespace yocto {

// Generates a ray from a camera for yimg::image plane coordinate uv and
// the lens coordinates luv.
static ray3f eval_camera(const camera_data& camera, const vec2f& uv) {
  // YOUR CODE GOES HERE
  
  // return object of type ray3f

  float d = camera.film;
  float camera_x = camera.film * (0.5 - uv.x);
  float camera_y = camera.film * (uv.y - 0.5);

  auto  q        = vec3f{camera_x, camera_y, camera.lens};
  auto e = vec3f{0};
  auto d_par        = normalize(-q - e);

  return ray3f{transform_point(camera.frame, e),
      transform_direction(camera.frame, d_par)};
}

}  // namespace yocto

// -----------------------------------------------------------------------------
// IMPLEMENTATION FOR PATH TRACING
// -----------------------------------------------------------------------------
namespace yocto {


// Raytrace renderer.
static vec4f shade_raytrace(const scene_data& scene, const bvh_scene& bvh,
    const ray3f& ray, int bounce, rng_state& rng,
    const raytrace_params& params) {
  // YOUR CODE GOES HERE
    /*
    
    
    --> implement a shader that simulates illumination for a variety of materials structured following the steps in the lecture notes
    --> get position, normal and texcoords; correct normals for lines
    --> get material values by multiply material constants and textures
    --> implement polished transmission, polished metals, rough metals, rough plastic, and matte shading in hte order described in the slides
    --> you can use any function from Yocto/Shading such as fresnel_schlick(), and the appropriate sampling functions sample_XXX()
    --> for matte, use the function sample_hemisphere_cos() and skip the normalization by 2 pi
    --> for rough metals, we will use a simpler version than the one used in the slides; we will choose a microfacet normal in an accurate manner and then implement reflection as if the metal was polished, with that normal
         - mnormal = sample_hemisphere_cospower(exponent, normal, rand2f(rng));
         - exponent = 2 / (roughness * roughness)
    --> for plastic surfaces, we will use the same manner to compute the normal
    --> and then apply a test with Fresnel that chooses whether to do reflection or diffuse
    
    */

  auto isec = intersect_bvh(bvh, scene, ray);
  if (!isec.hit) {
    auto env = eval_environment(scene, ray.d);
    return vec4f{env.x, env.y, env.z, 1};
  }

  auto& instance = scene.instances[isec.instance];
  auto& shape    = scene.shapes[instance.shape];
  auto& material = scene.materials[instance.material];
  
  vec3f emission;
  texture_data texture_emission;
  
  auto texcoord = eval_texcoord(shape, isec.element, isec.uv);

  
  if (material.emission_tex < 0) {
    emission = material.emission;
  } else {
    emission = rgba_to_rgb(rgb_to_rgba(material.color) * eval_texture(scene, material.emission_tex, texcoord, true));
  }

  // material constants
  auto metallic     = material.metallic;
  auto radiance     = emission;
  
  if (bounce >= params.bounces) return rgb_to_rgba(radiance);

  auto vec = vec3f{0.04, 0.04, 0.04};
  auto eta = reflectivity_to_eta(vec);

  auto normal = transform_direction(instance.frame, eval_normal(shape, isec.element, isec.uv));
  auto outgoing = -ray.d;


  // handling normals and lines
  if (!shape.lines.empty()) {
    // Tangent lines
    normal = orthonormalize(outgoing, normal);
  } else if (!shape.triangles.empty()) {
    // Flip the normal if normal and outgoing are in opposite directions
    if (dot(outgoing, normal) < 0) {
      normal = -normal;
    }
  }
  
  
  vec3f color;
  float roughness;
  texture_data texture_color;
  texture_data texture_roughness;

  
  if (material.color_tex < 0) {                     
    color = (material.color);                                   
  } else {
    color = rgba_to_rgb(rgb_to_rgba(material.color) * eval_texture(scene, material.color_tex, texcoord, true));
  }

  if (material.roughness_tex < 0) {
    roughness = material.roughness;
  } else {
    roughness = (material.roughness * eval_texture(scene, material.roughness_tex, texcoord, true)).x;
  }

  // Compute the position for the his frame
  auto position = transform_point(instance.frame, eval_position(shape, isec.element, isec.uv));

  auto color_tex = eval_texture(scene, material.color_tex, texcoord, true);

  auto opacity = material.opacity * color_tex.w;
  
  // handle opacity
  if ((rand1f(rng) < 1 - opacity)) {
    auto incoming = ray.d;
    return shade_raytrace(scene, bvh, ray3f{position, incoming}, bounce + 1, rng, params);
  }

 
  switch (material.type) {
    case material_type::glossy: {
      // handle rough plastic
      auto exponent = 2 / (roughness * roughness * roughness * roughness);
      auto incoming = sample_hemisphere_cospower(exponent, normal, rand2f(rng));

      auto vec = vec3f{0.04, 0.04, 0.04};
      if (rand1f(rng) < mean(fresnel_schlick(vec, incoming, outgoing))) {
        incoming = reflect(outgoing, incoming);
        radiance += xyz(shade_raytrace(
            scene, bvh, ray3f{position, incoming}, (bounce + 1), rng, params));
      } else {
        incoming = sample_hemisphere_cos(incoming, rand2f(rng));
        radiance += color *
                    xyz(shade_raytrace(scene, bvh, ray3f{position, incoming},
                        (bounce + 1), rng, params));
      }
      break;
    }
    case material_type::matte: {
      // handle diffuse for matte surfeces
      auto incoming = sample_hemisphere_cos(normal, rand2f(rng));
      // updating radiance
      radiance += color *
                  xyz(shade_raytrace(scene, bvh, ray3f{position, incoming},
                      (bounce + 1), rng, params));
      break;
    }
    case material_type::reflective: {
      if (!material.roughness) {
        //polished
        auto incoming = reflect(outgoing, normal);
        radiance += (fresnel_schlick(color, normal, outgoing)) * xyz(shade_raytrace(scene, bvh, ray3f{position, incoming}, bounce + 1, rng, params));
      } else {
        auto exponent = 2 / (roughness * roughness * roughness * roughness);
        auto mnormal  = sample_hemisphere_cospower(exponent, normal, rand2f(rng));

        auto incoming = reflect(outgoing, mnormal);
        radiance += (fresnel_schlick(color, mnormal, outgoing)) * xyz(shade_raytrace(scene, bvh, ray3f{position, incoming}, bounce + 1, rng, params));
      }

      break;
    }
    case material_type::transparent: {

      if (rand1f(rng) < mean(fresnel_schlick(vec3f{0.04, 0.04, 0.04}, normal, outgoing))) {
        //
        // incoming is obtained through reflect function
        //
        auto incoming = reflect(outgoing, normal);
        radiance += xyz(shade_raytrace(
            scene, bvh, {position, incoming}, bounce + 1, rng, params));
      } else {
        //
        // incoming is simply the opposite of the outgoing
        //
        // polished
        auto incoming = -outgoing;
        radiance += color *
                           xyz(shade_raytrace(scene, bvh, {position, incoming},
                               bounce + 1, rng, params));
      }
      break;
    }

  }
    
  return rgb_to_rgba(radiance);
}


static vec4f shade_refract(const scene_data& scene, const bvh_scene& bvh,
    const ray3f& ray, int bounce, rng_state& rng,
    const raytrace_params& params) {

  auto isec = intersect_bvh(bvh, scene, ray);
  if (!isec.hit) {
    auto env = eval_environment(scene, ray.d);
    return vec4f{env.x, env.y, env.z, 1};
  }

  auto& instance = scene.instances[isec.instance];
  auto& shape    = scene.shapes[instance.shape];
  auto& material = scene.materials[instance.material];

  vec3f        emission;
  texture_data texture_emission;

  auto texcoord = eval_texcoord(shape, isec.element, isec.uv);

  if (material.emission_tex < 0) {
    emission = material.emission;
  } else {

    emission = rgba_to_rgb(
        rgb_to_rgba(material.color) *
        eval_texture(scene, material.emission_tex, texcoord, true));
  }

  // material constants
  auto metallic = material.metallic;
  auto radiance = emission;

  if (bounce >= params.bounces) return rgb_to_rgba(radiance);

  auto vec = vec3f{0.04, 0.04, 0.04};
  auto eta = reflectivity_to_eta(vec);

  auto normal = transform_direction(
      instance.frame, eval_normal(shape, isec.element, isec.uv));
  auto outgoing = -ray.d;

  // handling normals and lines
  if (!shape.lines.empty()) {
    // Tangent lines
    normal = orthonormalize(outgoing, normal);
  } else if (!shape.triangles.empty()) {
    // Flip the normal if normal and outgoing are in opposite directions
    // correction normal& eta
    if (dot(outgoing, normal) < 0) {
      normal = -normal;
      eta    = 1 / eta;
    }
  }

  vec3f        color;
  float        roughness;
  texture_data texture_color;
  texture_data texture_roughness;

  if (material.color_tex < 0) {
    color = (material.color);
  } else {

    color = rgba_to_rgb(
        rgb_to_rgba(material.color) *
        eval_texture(scene, material.color_tex, texcoord, true));
  }

  if (material.roughness_tex < 0) {
    roughness = material.roughness;
  } else {

    roughness = (material.roughness *
                 eval_texture(scene, material.roughness_tex, texcoord, true))
                    .x;
  }

  auto position = transform_point(
      instance.frame, eval_position(shape, isec.element, isec.uv));

  auto color_tex = eval_texture(scene, material.color_tex, texcoord, true);

  auto opacity = material.opacity * color_tex.w;

  // handle opacity
  if (rand1f(rng) < 1 - opacity) {
    auto incoming = ray.d;
    return shade_raytrace(
        scene, bvh, ray3f{position, incoming}, bounce + 1, rng, params);
  }

  switch (material.type) {
    case material_type::glossy: {
      // handle rough plastic
      auto exponent = 2 / (roughness * roughness * roughness * roughness);
      auto incoming = sample_hemisphere_cospower(exponent, normal, rand2f(rng));

      auto vec = vec3f{0.04, 0.04, 0.04};
      if (rand1f(rng) < mean(fresnel_schlick(vec, incoming, outgoing))) {
        incoming = reflect(outgoing, incoming);
        radiance += xyz(shade_raytrace(
            scene, bvh, ray3f{position, incoming}, (bounce + 1), rng, params));
      } else {
        incoming = sample_hemisphere_cos(incoming, rand2f(rng));
        radiance += color *
                    xyz(shade_raytrace(scene, bvh, ray3f{position, incoming},
                        (bounce + 1), rng, params));
      }
      break;
    }
    case material_type::matte: {
      // handle diffuse for matte surfeces
      auto incoming = sample_hemisphere_cos(normal, rand2f(rng));
      // updating radiance
      radiance += color *
                  xyz(shade_raytrace(scene, bvh, ray3f{position, incoming},
                      (bounce + 1), rng, params));
      break;
    }
    case material_type::reflective: {
      if (!material.roughness) {
        // polished
        auto incoming = reflect(outgoing, normal);
        radiance += (fresnel_schlick(color, normal, outgoing)) * xyz(shade_raytrace(scene, bvh, ray3f{position, incoming}, bounce + 1, rng, params));
      } else {
        auto exponent = 2 / (roughness * roughness * roughness * roughness);
        auto mnormal  = sample_hemisphere_cospower(exponent, normal, rand2f(rng));

        auto incoming = reflect(outgoing, mnormal);
        radiance += (fresnel_schlick(color, mnormal, outgoing)) * xyz(shade_raytrace(scene, bvh, ray3f{position, incoming}, bounce + 1, rng, params));
      }

      break;
    }
    case material_type::transparent: {
      // refraction
      auto ior = reflectivity_to_eta(vec);
      auto eta = 1.f / mean(ior);


    if (rand1f(rng) < mean(fresnel_schlick(vec3f{0.04, 0.04, 0.04}, normal, outgoing))) {
          
        auto incoming = reflect(outgoing, normal);
        radiance += xyz(shade_raytrace(
            scene, bvh, {position, incoming}, bounce + 1, rng, params));
    } else {
        auto incoming = refract(outgoing, normal, eta);
                                    
        radiance += color *
                    xyz(shade_raytrace(scene, bvh, {position, incoming},
                                bounce + 1, rng, params));
    }
      
      break;
    }
  }

  return rgb_to_rgba(radiance);
}


static vec3f get_toon_magic(vec3f incoming, vec3f color, vec3f normal,
    vec3f outgoing, int _shadow, vec3f light_pos_wrld) {

  auto ambient_color = vec4f{0.4, 0.4, 0.4, 1};
  auto light_color   = vec4f{0.4, 0.4, 0.4, 1};
  double shadow      = 2;



  float light_ext_normal = dot(light_pos_wrld, normal);
  //shadow color cut
  float intensty_of_light = smoothstep(0.0, 0.1, light_ext_normal * shadow);


  auto light = intensty_of_light * _shadow;

  auto  specular_color = vec4f{0.9, 0.9, 0.9, 1};
  float glossness      = 32;

  auto  halfway            = normalize(light_pos_wrld + outgoing);
  float normal_dot_halfway = dot(normal, halfway);
  double specular_intensty = pow((normal_dot_halfway * intensty_of_light), (glossness * glossness));

  float spec_intensty_smth = smoothstep(0.005, 0.01, specular_intensty);
  auto  specular           = spec_intensty_smth * specular_color;


  auto  rim_dot   = 1 - dot(outgoing, normal);
  auto  rim_color = vec4f{1, 1, 1, 1};
  float rim_amnt  = 0.9;

  float rim_threshold = 0.1;
  double rim_intensty = rim_dot * pow(light_ext_normal, rim_threshold);
  rim_intensty        = smoothstep(rim_amnt - 0.01, rim_amnt + 0.01, rim_intensty);
  auto rim            = rim_intensty * rim_color;


  return color * xyz(light + ambient_color + rim + specular);
}


static vec4f own_shader(const scene_data& scene, const bvh_scene& bvh,
    const ray3f& ray, int bounce, rng_state& rng, const raytrace_params& params){


  auto isec = intersect_bvh(bvh, scene, ray);
  if (!isec.hit) {
    auto env = eval_environment(scene, ray.d);
    return vec4f{env.x, env.y, env.z, 1};
  }

  auto& instance = scene.instances[isec.instance];
  auto& shape    = scene.shapes[instance.shape];
  auto& material = scene.materials[instance.material];

  auto position = transform_point(instance.frame, eval_position(shape, isec.element, isec.uv));
  auto normal = transform_direction(
      instance.frame, eval_normal(shape, isec.element, isec.uv));
  auto outgoing = -ray.d;

  vec3f color;
  auto texcoord = eval_texcoord(shape, isec.element, isec.uv);
  if (material.color_tex < 0) {
    color = (material.color);
  } else {
    color = rgba_to_rgb(
        rgb_to_rgba(material.color) *
        eval_texture(scene, material.color_tex, texcoord, true));
  }


  auto radiance = color;

  if (!shape.lines.empty()) {
    // Tangent lines
    normal = orthonormalize(outgoing, normal);
  } else if (!shape.triangles.empty()) {
    // Flip the normal
    if (dot(outgoing, normal) < 0)
        normal = -normal;
  }


  if (bounce >= params.bounces) {
    return rgb_to_rgba(radiance);
  }

  auto incoming = sample_hemisphere(normal, rand2f(rng));


  // Now i'll check if there's intersection with the light position

  vec3f light_pos_wrld = {10.0, 16.0, 6.0};

  auto inshadow_isec = intersect_bvh(bvh, scene, {position, light_pos_wrld});

  auto _shadow = inshadow_isec.hit ? 0 : 1;

  radiance *= get_toon_magic(incoming, color, normal, outgoing, _shadow, light_pos_wrld);

  radiance += (pif) * (color)*abs(dot(normal, incoming)) * xyz(own_shader(scene, bvh, {position, incoming}, bounce + 1, rng, params));

  return rgb_to_rgba(radiance);
}




// Matte renderer.
static vec4f shade_matte(const scene_data& scene, const bvh_scene& bvh,
    const ray3f& ray, int bounce, rng_state& rng,
    const raytrace_params& params) {
  // YOUR CODE GOES HERE

  return {1, 0, 0, 1};
}

// Eyelight for quick previewing.
static vec4f shade_eyelight(const scene_data& scene, const bvh_scene& bvh,
    const ray3f& ray, int bounce, rng_state& rng,
    const raytrace_params& params) {
  // YOUR CODE GOES HERE

  auto isec = intersect_bvh(bvh, scene, ray);
  if (!isec.hit) return zero4f;

  auto& object = scene.instances[isec.instance];
  auto& shape  = scene.shapes[object.shape];

  auto normal = transform_direction(object.frame, eval_normal(shape, isec.element, isec.uv));

  normal = normal * 0.5f + 0.5f;

  auto shadde = abs(dot(normal, -ray.d));
  return rgb_to_rgba(scene.materials[object.material].color * shadde);
}

static vec4f shade_normal(const scene_data& scene, const bvh_scene& bvh,
    const ray3f& ray, int bounce, rng_state& rng,
    const raytrace_params& params) {
  // YOUR CODE GOES HERE

  auto isec = intersect_bvh(bvh, scene, ray);
  if (!isec.hit) return zero4f;

  auto& object = scene.instances[isec.instance];
  auto& shape  = scene.shapes[object.shape];

  auto normal = transform_direction(object.frame, eval_normal(shape, isec.element, isec.uv));

  normal = normal * 0.5f + 0.5f;

  return rgb_to_rgba(normal);
}

static vec4f shade_texcoord(const scene_data& scene, const bvh_scene& bvh,
    const ray3f& ray, int bounce, rng_state& rng,
    const raytrace_params& params) {
  // YOUR CODE GOES HERE
  auto isec = intersect_bvh(bvh, scene, ray);
  if (!isec.hit) return {0, 0, 0, 1};

  auto& object = scene.instances[isec.instance];
  auto& shape  = scene.shapes[object.shape];

  auto texture_coordinates = eval_texcoord(shape, isec.element, isec.uv);

  return {fmod(texture_coordinates.x, 1), fmod(texture_coordinates.y, 1), 0, 1};
}

static vec4f shade_color(const scene_data& scene, const bvh_scene& bvh,
    const ray3f& ray, int bounce, rng_state& rng,
    const raytrace_params& params) {
  // YOUR CODE GOES HERE

  auto intersection = intersect_bvh(bvh, scene, ray);
  if (!intersection.hit) return {0, 0, 0, 1};

  auto& object        = scene.instances[intersection.instance];
  auto& objcolor      = scene.materials[object.material];
  vec4f result;
  
  result.x = objcolor.color.x;
  result.y = objcolor.color.y;
  result.z = objcolor.color.z;
  result.w = objcolor.opacity;
  return result;
}

// Trace a single ray from the camera using the given algorithm.
using raytrace_shader_func = vec4f (*)(const scene_data& scene,
    const bvh_scene& bvh, const ray3f& ray, int bounce, rng_state& rng,
    const raytrace_params& params);
static raytrace_shader_func get_shader(const raytrace_params& params) {
  switch (params.shader) {
    case raytrace_shader_type::raytrace: return shade_raytrace;
    case raytrace_shader_type::matte: return shade_matte;
    case raytrace_shader_type::eyelight: return shade_eyelight;
    case raytrace_shader_type::normal: return shade_normal;
    case raytrace_shader_type::texcoord: return shade_texcoord;
    case raytrace_shader_type::color: return shade_color;
    case raytrace_shader_type::ownshader: return own_shader;
    case raytrace_shader_type::refractor: return shade_refract;
    default: {
      throw std::runtime_error("sampler unknown");
      return nullptr;
    }
  }
}

// Build the bvh acceleration structure.
bvh_scene make_bvh(const scene_data& scene, const raytrace_params& params) {
  return make_bvh(scene, false, false, params.noparallel);
}

// Init a sequence of random number generators.
raytrace_state make_state(
    const scene_data& scene, const raytrace_params& params) {
  auto& camera = scene.cameras[params.camera];
  auto  state  = raytrace_state{};
  if (camera.aspect >= 1) {
    state.width  = params.resolution;
    state.height = (int)round(params.resolution / camera.aspect);
  } else {
    state.height = params.resolution;
    state.width  = (int)round(params.resolution * camera.aspect);
  }
  state.samples = 0;
  state.image.assign(state.width * state.height, {0, 0, 0, 0});
  state.hits.assign(state.width * state.height, 0);
  state.rngs.assign(state.width * state.height, {});
  auto rng_ = make_rng(1301081);
  for (auto& rng : state.rngs) {
    rng = make_rng(961748941ull, rand1i(rng_, 1 << 31) / 2 + 1);
  }
  return state;
}

void trace_sample(raytrace_state& state, const scene_data& scene,
    const bvh_scene& bvh, vec2i& ijdx, const raytrace_params& params) {



}

void render_sample(raytrace_state& state, const scene_data& scene,
    const bvh_scene& bvh, const raytrace_params& params, int i, int j) {

  auto& camera       = scene.cameras[params.camera];
  auto  shader       = get_shader(params);
  auto  image_width  = state.width;
  auto  image_height = state.height;
  auto  clamp_value  = 100000;  // CLAMP VALUE (I'VE CHOOSE IT)

  // I'LL TAKE ONLY ONE PIXEL

  int idx = i + (image_width * j);  //+i;

  auto puv = rand2f(state.rngs[idx]);
  auto luv = rand2f(state.rngs[idx]);
  auto p   = vec2f{(float)i, (float)j} + puv;
  auto uv  = vec2f{p.x / image_width, p.y / image_height};

  // now i create the ray
  auto ray = eval_camera(camera, uv, luv);


  auto color = shader(scene, bvh, ray, 0, state.rngs[idx], params);

  if (length(xyz(color)) > clamp_value)  // CLAMP VALUE (I'VE CHOOSE IT)
    color = normalize(color) * clamp_value;

  state.image[idx] += color;
}

// Progressively compute an image by calling trace_samples multiple times.
void raytrace_samples(raytrace_state& state, const scene_data& scene,
    const bvh_scene& bvh, const raytrace_params& params) {

   
  if (state.samples >= params.samples) return;
  // YOUR CODE GOES HERE

  
  auto image_width  = state.width;
  auto image_height = state.height;


  if (params.noparallel) {
    for (auto j = 0; j < image_height; j++) {
      for (auto i = 0; i < image_width; i++) {
        render_sample(state, scene, bvh, params, i, j);
      }
    }
  } else {
      parallel_for(state.width, state.height, [&](int i, int j){
          render_sample(state, scene, bvh, params, i, j);
      });

  }

  
  state.samples += 1;
}

// Check image type
static void check_image(
    const color_image& image, int width, int height, bool linear) {
  if (image.width != width || image.height != height)
    throw std::invalid_argument{"image should have the same size"};
  if (image.linear != linear)
    throw std::invalid_argument{
        linear ? "expected linear image" : "expected srgb image"};
}

// Get resulting render
color_image get_render(const raytrace_state& state) {
  auto image = make_image(state.width, state.height, true);
  get_render(image, state);
  return image;
}
void get_render(color_image& image, const raytrace_state& state) {
  check_image(image, state.width, state.height, true);
  auto scale = 1.0f / (float)state.samples;
  for (auto idx = 0; idx < state.width * state.height; idx++) {
    image.pixels[idx] = state.image[idx]*scale;
  }
}

}  // namespace yocto
