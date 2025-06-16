import threading
import math
import traceback
import comfy.samplers

_original_sampling_function = comfy.samplers.sampling_function
_sampling_function_patched = False

thread_local_data = threading.local()

def set_active_cfg_info(info_dict):
    thread_local_data.active_cfg_info = info_dict

def get_active_cfg_info():
    return getattr(thread_local_data, 'active_cfg_info', None)

def clear_active_cfg_info():
    if hasattr(thread_local_data, 'active_cfg_info'):
        del thread_local_data.active_cfg_info

def apply_sampling_patch():
    global _sampling_function_patched, _original_sampling_function
    if not _sampling_function_patched:
        _original_sampling_function = comfy.samplers.sampling_function
        comfy.samplers.sampling_function = patched_sampling_function
        _sampling_function_patched = True
        print("[CustomCFG Patcher] comfy.samplers.sampling_function patched successfully.")
        return True
    return False

def remove_sampling_patch():
    global _sampling_function_patched, _original_sampling_function
    if _sampling_function_patched:
        comfy.samplers.sampling_function = _original_sampling_function
        _sampling_function_patched = False
        print("[CustomCFG Patcher] Original comfy.samplers.sampling_function restored.")
        return True
    return False

def find_sigma_index(sigma_value, sigmas_tensor_or_list):
    if sigmas_tensor_or_list is None or len(sigmas_tensor_or_list) == 0:
        print(f"[CustomCFG Helper] Warning: sigmas is empty or None. Defaulting to step 0 for sigma {sigma_value:.4f}.")
        return 0

    sigmas_list = sigmas_tensor_or_list.tolist() if hasattr(sigmas_tensor_or_list, 'tolist') else list(sigmas_tensor_or_list)
    
    if not sigmas_list:
        print(f"[CustomCFG Helper] Warning: sigmas became an empty list after conversion. Defaulting to step 0 for sigma {sigma_value:.4f}.")
        return 0

    if len(sigmas_list) == 1:
        return 0

    for i in range(len(sigmas_list) - 1):
        s_i = sigmas_list[i]
        s_i_plus_1 = sigmas_list[i+1]
        if math.isclose(s_i, sigma_value):
            return i
        if (s_i_plus_1 < s_i and s_i_plus_1 <= sigma_value < s_i) or \
           (s_i_plus_1 > s_i and s_i <= sigma_value < s_i_plus_1) or \
           (math.isclose(s_i_plus_1, s_i) and math.isclose(sigma_value, s_i)):
            return i
    
    if math.isclose(sigmas_list[-1], sigma_value):
        return len(sigmas_list) - 1

    print(f"[CustomCFG Helper] Warning: Sigma {sigma_value:.4f} not in ranges of sigmas (len {len(sigmas_list)}, "
          f"first: {sigmas_list[0]:.4f}, last: {sigmas_list[-1]:.4f}). Approximating.")
    
    first_sigma, last_sigma = sigmas_list[0], sigmas_list[-1]
    is_descending = first_sigma > last_sigma

    if is_descending:
        if sigma_value > first_sigma and not math.isclose(sigma_value, first_sigma): return 0
        if sigma_value < last_sigma and not math.isclose(sigma_value, last_sigma): return len(sigmas_list) - 1
    else: 
        if sigma_value < first_sigma and not math.isclose(sigma_value, first_sigma): return 0
        if sigma_value > last_sigma and not math.isclose(sigma_value, last_sigma): return len(sigmas_list) - 1
        
    closest_index = min(range(len(sigmas_list)), key=lambda i: abs(sigmas_list[i] - sigma_value))
    print(f"[CustomCFG Helper] Defaulting to closest sigma index: {closest_index} for sigma {sigma_value:.4f}.")
    return closest_index

def patched_sampling_function(model, x, timestep, uncond, cond, cond_scale, model_options={}, seed=None):
    active_cfg_info = get_active_cfg_info()
    cfg_to_use = cond_scale
    force_skip_uncond_this_step = False
    step_index = -1
    sigmas_for_search_in_patch = None
    if active_cfg_info:
        uses_call_counter = active_cfg_info.get("uses_call_counter_for_step", False)
        current_sigma_value_for_log = timestep[0].item() if hasattr(timestep, 'item') and not isinstance(timestep, (float, int)) else timestep
        if isinstance(current_sigma_value_for_log, list): current_sigma_value_for_log = current_sigma_value_for_log[0]


        if uses_call_counter:
            active_cfg_info["call_counter"] = active_cfg_info.get("call_counter", -1) + 1
            step_index = active_cfg_info["call_counter"]
        else:
            current_sigma_value = current_sigma_value_for_log 

            sigmas_from_node_input = active_cfg_info.get("sigmas_for_step_lookup")
            sigmas_from_options_transformer = model_options.get("transformer_options", {}).get("sample_sigmas")
            sigmas_from_options_direct = model_options.get("sigmas")

            if sigmas_from_node_input is not None and \
               (not hasattr(sigmas_from_node_input, 'numel') or sigmas_from_node_input.numel() > 0):
                sigmas_for_search_in_patch = sigmas_from_node_input
            elif sigmas_from_options_transformer is not None and \
                 (not hasattr(sigmas_from_options_transformer, 'numel') or sigmas_from_options_transformer.numel() > 0):
                sigmas_for_search_in_patch = sigmas_from_options_transformer
            elif sigmas_from_options_direct is not None and \
                 (not hasattr(sigmas_from_options_direct, 'numel') or sigmas_from_options_direct.numel() > 0):
                sigmas_for_search_in_patch = sigmas_from_options_direct
            
            if sigmas_for_search_in_patch is not None and len(sigmas_for_search_in_patch) > 0:
                step_index = find_sigma_index(current_sigma_value, sigmas_for_search_in_patch)
            else:
                print(f"[CustomCFG Patched] Warning: No valid sigmas found for step lookup (sigma: {current_sigma_value:.4f}). Default CFG will be used.")

        if step_index != -1:
            try:
                scheduled_cfgs = active_cfg_info.get("step_cfgs", [])
                scheduled_skips = active_cfg_info.get("step_skip_unconds", [])
                
                if 0 <= step_index < len(scheduled_cfgs):
                    cfg_to_use = scheduled_cfgs[step_index]
                    if scheduled_skips and 0 <= step_index < len(scheduled_skips):
                        force_skip_uncond_this_step = scheduled_skips[step_index]
                elif scheduled_cfgs:
                    cfg_to_use = scheduled_cfgs[-1] 
                    if scheduled_skips:
                        force_skip_uncond_this_step = scheduled_skips[-1]
                else:
                    print(f"[CustomCFG Patched] Warning: step_index {step_index} (sigma {current_sigma_value_for_log:.4f}) "
                          f"but no CFGs scheduled. Using default cond_scale {cond_scale:.2f}.")

                total_steps_in_schedule = 0
                if uses_call_counter:
                    total_steps_in_schedule = active_cfg_info.get("total_steps_scheduled_by_override", 0)
                elif sigmas_for_search_in_patch is not None:
                    total_steps_in_schedule = len(sigmas_for_search_in_patch)

                if total_steps_in_schedule > 0 and step_index >= (total_steps_in_schedule - 1):
                    if not active_cfg_info.get("schedule_completed", False):
                        active_cfg_info["schedule_completed"] = True

            except Exception as e:
                print(f"[CustomCFG Patched] Error applying CFG schedule for sigma {current_sigma_value_for_log:.4f} (step_index {step_index}): {e}")
                traceback.print_exc()
    
    uncond_to_process = uncond
    disable_cfg1_optimization = model_options.get("disable_cfg1_optimization", False)

    if force_skip_uncond_this_step or (math.isclose(cfg_to_use, 1.0) and not disable_cfg1_optimization):
        uncond_to_process = None
    
    try:
        out_cond_batch = comfy.samplers.calc_cond_batch(model, [cond, uncond_to_process], x, timestep, model_options)
        args_pre_cfg = {
            "conds": [cond, uncond_to_process], "conds_out": out_cond_batch, "cond_scale": cfg_to_use,
            "timestep": timestep, "input": x, "sigma": timestep, 
            "model": model, "model_options": model_options,
        }
        for fn in model_options.get("sampler_pre_cfg_function", []):
            out_cond_batch = fn(args_pre_cfg)
        return comfy.samplers.cfg_function(model, out_cond_batch[0], out_cond_batch[1], cfg_to_use, x, timestep, model_options, cond, uncond_to_process)
    except Exception as e:
        print(f"[CustomCFG Patched] Critical error in patched_sampling_function: {e}")
        traceback.print_exc()
        raise

class CustomCFGSchedule:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "enabled": ("BOOLEAN", {"default": True, "label_on": "Enabled", "label_off": "Disabled"}),
                "initial_cfg": ("FLOAT", {"default": 7.0, "min": -100.0, "max": 100.0, "step": 0.01}),
                "cfg_schedule_points": ("STRING", {"multiline": True, "default": ""}),
                "schedule_loop_length": ("INT", {"default": 0, "min": 0, "max": 1000, "step": 1, "label": "Schedule Loop Length (0=No Loop)"}),
                "interpolate": ("BOOLEAN", {"default": False, "label_on": "Interpolate CFG", "label_off": "Stepped CFG"}),
                "allow_overshoot_and_trim": ("BOOLEAN", {"default": False, "label_on": "Overshoot & Trim", "label_off": "Clamp to Max Steps", "label": "Schedule Point Behavior"}),
            }, 
            "optional": {
                "sigmas": ("SIGMAS",),
                "total_steps_override": ("INT", {"default": 0, "min": 0, "max": 1000, "step": 1, "label": "Total Steps (if no SIGMAS input, 0=disabled)"}),
                "passthrough_model": ("MODEL",)
            }
        }

    RETURN_TYPES = ("STRING", "MODEL", "SIGMAS")
    RETURN_NAMES = ("info", "passthrough_model_out", "sigmas_out")
    FUNCTION = "apply_schedule"
    CATEGORY = "tksw_node"
    OUTPUT_NODE = True 

    def __init__(self):
        self._last_resolved_points_for_display = []

    def _parse_schedule_points(self, schedule_text):
        schedule_list = []
        if not schedule_text or not schedule_text.strip():
            return []
        
        normalized_text = schedule_text.replace('\r\n', '\n').replace('\r', '\n')
        raw_entries = [entry for item in normalized_text.split(',') for entry in item.split('\n')]

        for pair_raw in raw_entries:
            pair = pair_raw.strip()
            if not pair or pair.startswith("#"): continue
            
            parts = pair.split(':', 2)
            if len(parts) < 2:
                print(f"[CustomCFG Node] Warning: Invalid schedule pair format: '{pair}'. Skipping.")
                continue

            target_step_str, cfg_str = parts[0].strip(), parts[1].strip()
            skip_uncond_flag = False
            if len(parts) == 3:
                skip_flag_str = parts[2].strip().lower()
                if skip_flag_str in {"s", "skip", "true", "1", "yes"}: skip_uncond_flag = True
                elif skip_flag_str not in {"", "ns", "noskip", "false", "0", "no"}:
                    print(f"[CustomCFG Node] Warning: Unknown skip flag '{parts[2].strip()}' in '{pair}'. Defaulting to 'no skip'.")
            
            if not target_step_str or not cfg_str:
                print(f"[CustomCFG Node] Warning: Empty target step or CFG value in '{pair}'. Skipping.")
                continue
            try:
                cfg = float(cfg_str)
            except ValueError:
                print(f"[CustomCFG Node] Warning: Could not parse CFG value in '{pair}'. Skipping.")
                continue
            schedule_list.append((target_step_str, cfg, skip_uncond_flag))
        return schedule_list

    def _resolve_schedule_point(self, target_step_str, total_steps_max_idx, loop_length_count, allow_overshoot, original_pair_info=""):
        target_step_str_lower = target_step_str.lower()
        
        if target_step_str_lower.startswith(('p', 'l')):
            prefix = target_step_str_lower[0]
            value_str = target_step_str[1:]
            try:
                percentage = float(value_str)
            except ValueError:
                print(f"[CustomCFG Node] Warning: Could not parse percentage '{value_str}' from '{target_step_str}' {original_pair_info}. Skipping.")
                return None

            if not (math.isclose(percentage, 0.0) or math.isclose(percentage, 1.0) or (0.0 < percentage < 1.0)):
                print(f"[CustomCFG Node] Warning: Percentage {percentage:.3f} from '{target_step_str}' {original_pair_info} "
                      f"is outside 0.0-1.0 range. Skipping.")
                return None
            
            base_value = 0
            if prefix == 'p': 
                base_value = total_steps_max_idx if total_steps_max_idx >= 0 else 0
            elif prefix == 'l': 
                if loop_length_count > 0:
                    base_value = (loop_length_count - 1)
                else:
                    print(f"[CustomCFG Node] Info: 'L' prefix in {original_pair_info} requires positive loop_length. Skipping.")
                    return None
            
            resolved_step = round(percentage * base_value)
            resolved_step = max(0, resolved_step) 

            if not allow_overshoot:
                if prefix == 'p' and total_steps_max_idx >=0:
                    resolved_step = min(resolved_step, total_steps_max_idx)
                elif prefix == 'l' and loop_length_count > 0:
                    resolved_step = min(resolved_step, loop_length_count - 1)
            return resolved_step
        else: 
            try:
                step = int(target_step_str)
                step = max(0, step) 
                return step
            except ValueError:
                print(f"[CustomCFG Node] Warning: Could not parse absolute step '{target_step_str}' {original_pair_info}. Skipping.")
                return None

    def _expand_schedule_to_steps(self, raw_point_schedule, initial_cfg_value, total_steps_max_idx, 
                                  interpolate=False, loop_length_count=0, allow_overshoot=False):
        self._last_resolved_points_for_display = [] 
        num_values_needed = total_steps_max_idx + 1 if total_steps_max_idx >= 0 else 0

        if num_values_needed == 0:
            return [], []

        resolved_points = []
        temp_display_map = {} 

        for original_target_str, cfg, skip_flag in (raw_point_schedule or []):
            original_pair_info = f"'{original_target_str}:{cfg}{':s' if skip_flag else ''}'"
            resolved_step = self._resolve_schedule_point(original_target_str, total_steps_max_idx, loop_length_count, allow_overshoot, original_pair_info)
            if resolved_step is not None:
                resolved_points.append((resolved_step, cfg, skip_flag))
                temp_display_map[(original_target_str, cfg, skip_flag)] = resolved_step
        
        if not any(p[0] == 0 for p in resolved_points):
            resolved_points.append((0, initial_cfg_value, False))
            temp_display_map[("0 (initial_cfg)", initial_cfg_value, False)] = 0
        
        if not resolved_points:
            print(f"[CustomCFG Node] Warning: No valid schedule points. Using initial_cfg ({initial_cfg_value}) for all {num_values_needed} values.")
            return [initial_cfg_value] * num_values_needed, [False] * num_values_needed

        resolved_points.sort(key=lambda x: x[0])
        
        sorted_display_items = sorted(temp_display_map.items(), key=lambda item: (item[1], str(item[0][0])))
        for (orig_spec_tuple, res_step) in sorted_display_items:
            orig_str, cfg_val, skip_val = orig_spec_tuple
            self._last_resolved_points_for_display.append((res_step, cfg_val, skip_val, str(orig_str)))
        
        effective_schedule = resolved_points
        if loop_length_count > 0 and not allow_overshoot:
            effective_schedule = [p for p in resolved_points if p[0] < loop_length_count]
            if not effective_schedule:
                initial_point = next((p for p in resolved_points if p[0] == 0), None)
                effective_schedule = [initial_point] if initial_point else [(0, initial_cfg_value, False)]
        
        if not effective_schedule: 
            print(f"[CustomCFG Node] Critical Error: Effective schedule became empty. Using initial_cfg for {num_values_needed} values.")
            return [initial_cfg_value] * num_values_needed, [False] * num_values_needed

        step_cfgs = [0.0] * num_values_needed
        step_skip_unconds = [False] * num_values_needed

        for i in range(num_values_needed):
            current_target_step = i
            reference_step = current_target_step % loop_length_count if loop_length_count > 0 else current_target_step

            cfg_val_for_step = effective_schedule[0][1]
            skip_val_for_step = effective_schedule[0][2]

            if not interpolate:
                for p_target_s, p_cfg, p_skip in effective_schedule:
                    if p_target_s <= reference_step:
                        cfg_val_for_step = p_cfg
                        skip_val_for_step = p_skip
                    else: break 
            else: 
                seg_start_step, seg_start_cfg, seg_start_skip = effective_schedule[0]
                seg_end_step, seg_end_cfg = seg_start_step, seg_start_cfg

                if reference_step >= effective_schedule[0][0]:
                    for p_idx, (p_t_step, p_t_cfg, p_t_skip) in enumerate(effective_schedule):
                        if p_t_step <= reference_step:
                            seg_start_step, seg_start_cfg, seg_start_skip = p_t_step, p_t_cfg, p_t_skip
                            if p_idx + 1 < len(effective_schedule):
                                seg_end_step, seg_end_cfg = effective_schedule[p_idx+1][0], effective_schedule[p_idx+1][1]
                            else: 
                                seg_end_step, seg_end_cfg = p_t_step, p_t_cfg 
                        else: 
                            seg_end_step, seg_end_cfg = p_t_step, p_t_cfg
                            break 
                
                denominator = float(seg_end_step - seg_start_step)
                if denominator > 1e-6: 
                    interp_t = (reference_step - seg_start_step) / denominator
                    interp_t = max(0.0, min(1.0, interp_t)) 
                    cfg_val_for_step = seg_start_cfg + (seg_end_cfg - seg_start_cfg) * interp_t
                else: 
                    cfg_val_for_step = seg_start_cfg
                skip_val_for_step = seg_start_skip
            
            step_cfgs[i] = cfg_val_for_step
            step_skip_unconds[i] = skip_val_for_step
            
        return step_cfgs, step_skip_unconds

    def apply_schedule(self, enabled, initial_cfg, cfg_schedule_points, schedule_loop_length, 
                       interpolate, allow_overshoot_and_trim, 
                       sigmas=None, total_steps_override=0, passthrough_model=None):
        info_lines = []
        node_class_name = self.__class__.__name__
        
        previous_active_info = get_active_cfg_info() 
        clear_active_cfg_info() 

        if previous_active_info:
            was_interrupted = not previous_active_info.get("schedule_completed", True)
            patched_by_this_type = previous_active_info.get("patch_applied_by_this_node_type") == node_class_name
            if was_interrupted and patched_by_this_type and not enabled:
                print(f"[{node_class_name}] Disabled after interruption. Cleaning up patch.")
                if remove_sampling_patch(): info_lines.append(f"CustomCFG ({node_class_name}): Disabled (cleaned up patch)")

        if not enabled:
            info_lines.insert(0, f"CustomCFG ({node_class_name}): Disabled")
            print(f"[{node_class_name}] CustomCFG is disabled.")
            if previous_active_info and previous_active_info.get("patch_applied_by_this_node_type") == node_class_name:
                if _sampling_function_patched and remove_sampling_patch():
                     info_lines[0] += " (Removed own patch)"
            return ("\n".join(info_lines), passthrough_model, sigmas)

        num_schedule_values = 0
        total_steps_max_idx = -1
        uses_call_counter_for_step = False
        sigmas_for_step_lookup_in_patch = None
        step_source_info = ""

        if sigmas is not None and hasattr(sigmas, 'shape') and sigmas.shape[0] > 0:
            num_schedule_values = sigmas.shape[0]
            sigmas_for_step_lookup_in_patch = sigmas
            step_source_info = f"SIGMAS input ({num_schedule_values} steps)"
        elif total_steps_override > 0:
            num_schedule_values = total_steps_override
            uses_call_counter_for_step = True
            step_source_info = f"Total Steps Override ({num_schedule_values} steps, using call counter)"
        else:
            msg = f"CustomCFG ({node_class_name}): Enabled (Error: Neither SIGMAS input nor valid Total Steps Override provided)"
            print(f"[{node_class_name}] Error: SIGMAS input missing/invalid and Total Steps Override is not positive.")
            if _sampling_function_patched and previous_active_info and \
               previous_active_info.get("patch_applied_by_this_node_type") == node_class_name and \
               remove_sampling_patch():
                msg += "\n(Removed own patch)"
            return (msg, passthrough_model, sigmas)

        total_steps_max_idx = num_schedule_values - 1

        if num_schedule_values == 0 :
            msg = f"CustomCFG ({node_class_name}): Enabled (No sampling steps defined)"
            print(f"[{node_class_name}] No sampling steps determined from inputs.")
            if _sampling_function_patched and previous_active_info and \
               previous_active_info.get("patch_applied_by_this_node_type") == node_class_name and \
               remove_sampling_patch():
                msg += "\n(Removed own patch)"
            return (msg, passthrough_model, sigmas)

        parsed_point_schedule = self._parse_schedule_points(cfg_schedule_points)
        
        effective_loop_length = schedule_loop_length
        loop_info_str = f"Looping: Disabled (Len {schedule_loop_length})"
        overshoot_mode_str = "Overshoot & Trim" if allow_overshoot_and_trim else "Clamp to Max"
        
        if schedule_loop_length > 0:
            if not allow_overshoot_and_trim and schedule_loop_length > num_schedule_values : 
                print(f"[{node_class_name}] Warning: Loop Length ({schedule_loop_length}) > Total Values ({num_schedule_values}) "
                      f"and not Overshoot mode. Loop ineffective.")
                loop_info_str =f"Looping: Disabled (Len {schedule_loop_length} > {num_schedule_values} values, not overshoot)"
                effective_loop_length = 0 
            else:
                loop_info_str = f"Looping: Enabled (Length: {schedule_loop_length} steps)"
        
        expanded_step_cfgs, expanded_step_skip_unconds = self._expand_schedule_to_steps(
            parsed_point_schedule, initial_cfg, total_steps_max_idx, 
            interpolate, effective_loop_length, allow_overshoot_and_trim
        )
        
        if not expanded_step_cfgs or len(expanded_step_cfgs) != num_schedule_values:
            msg = f"CustomCFG ({node_class_name}): Enabled (Error: Failed to expand schedule to {num_schedule_values} values. Check console.)"
            print(f"[{node_class_name}] Error: Schedule expansion failed. Expected {num_schedule_values} CFG values, got {len(expanded_step_cfgs)}.")
            return (msg, passthrough_model, sigmas)

        new_active_info = {
            "step_cfgs": expanded_step_cfgs,
            "step_skip_unconds": expanded_step_skip_unconds,
            "total_steps_max_idx_for_schedule_logic": total_steps_max_idx,
            "schedule_completed": False,
            "patch_applied_by_this_node_type": node_class_name,
            "uses_call_counter_for_step": uses_call_counter_for_step,
            "sigmas_for_step_lookup": sigmas_for_step_lookup_in_patch
        }
        if uses_call_counter_for_step:
            new_active_info["call_counter"] = -1
            new_active_info["total_steps_scheduled_by_override"] = num_schedule_values


        patch_newly_applied = apply_sampling_patch()
        set_active_cfg_info(new_active_info)

        info_lines.append(f"CustomCFG ({node_class_name}): Enabled")
        info_lines.append(f"Step Source: {step_source_info}")
        info_lines.append(f"Initial CFG: {initial_cfg:.2f}")
        info_lines.append(f"Mode: {'Interpolated' if interpolate else 'Stepped'}")
        info_lines.append(f"Point Behavior: {overshoot_mode_str}")
        info_lines.append(loop_info_str)
        info_lines.append(f"Schedule Values: {num_schedule_values} (for steps 0 to {total_steps_max_idx})")
        
        first_sched_line = cfg_schedule_points.splitlines()[0] if cfg_schedule_points else "N/A"
        info_lines.append(f"Input Str: '{first_sched_line}{'...' if len(cfg_schedule_points.splitlines()) > 1 else ''}'")

        resolved_pts_display = []
        display_basis = self._last_resolved_points_for_display
        if effective_loop_length > 0 and not allow_overshoot_and_trim:
            filtered_basis = [pt for pt in display_basis if pt[0] < effective_loop_length]
            if not filtered_basis and display_basis: 
                 initial_pt_display = next((p for p in display_basis if p[0]==0), None)
                 if initial_pt_display: filtered_basis = [initial_pt_display]
            display_basis = filtered_basis if filtered_basis else display_basis

        for res_step, p_cfg, p_skip, orig_str in display_basis:
            orig_cleaned = orig_str.replace("(initial_cfg)",f"(initial:{initial_cfg:.1f})")
            resolved_pts_display.append(f"{orig_cleaned} -> {res_step}:{p_cfg:.1f}{':s' if p_skip else ''}")
        if not resolved_pts_display and initial_cfg is not None: 
            resolved_pts_display.append(f"0 (initial):{initial_cfg:.1f} (Default)")
        info_lines.append(f"Resolved Basis Points: [{', '.join(resolved_pts_display)}]")

        display_limit = min(20, num_schedule_values)
        cfgs_formatted = [f"{cfgs:.2f}{':s' if skips else ''}" 
                          for cfgs, skips in zip(expanded_step_cfgs[:display_limit], expanded_step_skip_unconds[:display_limit])]
        cfgs_str = ", ".join(cfgs_formatted)
        if display_limit < num_schedule_values: cfgs_str += f", ... ({num_schedule_values - display_limit} more)"
        info_lines.append(f"Expanded CFGs (first {display_limit}): [{cfgs_str}]")

        patch_status_msg = "Patch Status: "
        if patch_newly_applied: patch_status_msg += "Newly applied."
        elif _sampling_function_patched: patch_status_msg += "Already active. Schedule (re)set."
        else: patch_status_msg += "Error - Inactive after attempt."
        info_lines.append(patch_status_msg)
        
        print(f"[{node_class_name}] Schedule set. ({step_source_info}, {'Interpolated' if interpolate else 'Stepped'}, "
              f"{num_schedule_values} values, {loop_info_str.lower()}, {overshoot_mode_str.lower()})")
        
        return ("\n".join(info_lines), passthrough_model, sigmas)

    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
        return float("NaN")