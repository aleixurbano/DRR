
def get_robot_plan(args, step=None, with_obs=False):
    if args.ablation_type == 0 and args.audio_ver == 0:
        summary_file_name = 'state_summary_L2_wo_sound.txt'
    elif args.ablation_type == 5:
        summary_file_name = 'state_summary_L2_BLIP2.txt'
    else:
        summary_file_name = 'state_summary_L2.txt'
    with open('real_world/state_summary/{}/{}'.format(args.folder_name, summary_file_name), 'r') as f:
        L2_captions = f.readlines()

    if args.ablation_type == 0 and args.audio_ver == 0:
        summary_file_name = 'state_summary_L1_wo_sound.txt'
    elif args.ablation_type == 5:
        summary_file_name = 'state_summary_L1_BLIP2.txt'
    else:
        summary_file_name = 'state_summary_L1.txt'
    with open('real_world/state_summary/{}/{}'.format(args.folder_name, summary_file_name), 'r') as f:
        L1_captions = f.readlines()

    if with_obs is False:
        captions = L2_captions
    else:
        if args.ablation_type in [3]:
            captions = L2_captions
        elif args.ablation_type in [0, 1, 5]:
            captions = L1_captions

    robot_plan = ""
    for caption in captions:
        if step is not None and step in caption:
            break
        if with_obs:
            robot_plan += caption
        else:
            robot_plan += caption[:caption.find("Visual observation")-1] + "\n"
    return robot_plan
