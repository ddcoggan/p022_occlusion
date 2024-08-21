# /usr/bin/python
# Created by David Coggan on 2023 06 21

def registration_and_ROI_masks(subjects, method):

    for subject in subjects:

        # convert anatomical to nifti (final preprocessed anatomical)
        fs_dir = f'derivatives/freesurfer/sub-{subject}'
        mgz = f'{fs_dir}/mri/T1.mgz'
        nii = f'{fs_dir}/mri/T1.nii'
        if not op.isfile(nii):
            os.system(f'mri_convert {mgz} {nii}')

        # extract brain
        nii_brain = f'{fs_dir}/mri/T1_brain.nii.gz'
        if not op.isfile(nii_brain):
            os.system(f'bet {nii} {nii_brain}')

        # convert anatomical to nifti (original anatomical)
        mgz = f'{fs_dir}/mri/orig/001.mgz'
        nii = f'{fs_dir}/mri/orig/001.nii'
        if not op.isfile(nii):
            os.system(f'mri_convert {mgz} {nii}')

        # extract brain
        nii_brain = f'{fs_dir}/mri/orig/001_brain.nii.gz'
        if not op.isfile(nii_brain):
            os.system(f'bet {nii} {nii_brain}')

        # create transformation matrix from standard space to anatomical space
        xform_std_to_anat = f'{fs_dir}/mri/transforms/reg.mni152.2mm.lta'
        if not op.isfile(xform_std_to_anat):
            os.system(f'mni152reg --s sub-{subject}')

        # get retinotopy estimates
        from utils.get_wang_atlas import get_wang_atlas
        get_wang_atlas(f'sub-{subject}')  # automatically skips if already performed

        reg_dir = f'derivatives/registrations/sub-{subject}'
        os.makedirs(reg_dir, exist_ok=True)

        ref_func = glob.glob(f'derivatives/FEAT/sub-{subject}/*.gfeat/cope1.feat/example_func.nii.gz')[0]
        ref_anat = f'{fs_dir}/mri/orig/001.nii'
        ref_anat_brain = f'{fs_dir}/mri/orig/001_brain.nii.gz'

        # create transformation matrix between anatomical space and func space
        if method == 'FSL':

            xform_func_to_anat = f'{reg_dir}/func_to_anat.mat'
            if not op.isfile(xform_func_to_anat):
                os.system(
                    f'epi_reg --epi={ref_func} --t1={ref_anat} --t1brain={ref_anat_brain} --out={xform_func_to_anat[:-4]}')  # BBR
                # os.system(f'flirt -in {ref_anat_brain} -ref {ref_func} -omat {xform_anat_to_func}')  # linear search
            xform_anat_to_func = f'{reg_dir}/anat_to_func.mat'
            if not op.isfile(xform_anat_to_func):
                os.system(f'convert_xfm -omat {xform_anat_to_func} -inverse {xform_func_to_anat}')

        elif method == 'freesurfer':

            xform_func_to_anat = f'{reg_dir}/func_to_anat.lta'
            if not op.isfile(xform_func_to_anat):
                os.system(
                    f'bbregister --s sub-{subject} --mov {ref_func} --init-fsl --reg {xform_func_to_anat} --bold')


        # create ROI masks
        mask_dir = f'derivatives/masks/sub-{subject}'
        for mask_label, mask in fMRI.region_mapping.items():


            # std space to anat space
            mask_path_std = glob.glob(f'{fMRI.mask_dir_std}/*/{mask}.nii.gz')[0]
            mask_path_anat = f'{mask_dir}/anat_space/{mask_label}.nii.gz'
            if not op.isfile(mask_path_anat):
                os.makedirs(op.dirname(mask_path_anat), exist_ok=True)
                os.system(
                    f'mri_vol2vol --mov {mask_path_std} --targ {ref_anat} --lta {xform_std_to_anat} --o {mask_path_anat} --nearest')

            # anat space to func space
            if method == 'FSL':
                mask_path_func = f'{mask_dir}/func_space/{mask_label}.nii.gz'
                if not op.isfile(mask_path_func):
                    os.makedirs(op.dirname(mask_path_func), exist_ok=True)
                    os.system(
                        f'flirt -in {mask_path_anat} -ref {ref_func} -applyxfm -init {xform_anat_to_func} -out {mask_path_func} -interp nearestneighbour')

            elif method == 'freesurfer':
                mask_path_func = f'derivatives/masks/sub-{subject}/func_space/{mask_label}.nii.gz'
                if not op.isfile(mask_path_func):
                    os.makedirs(op.dirname(mask_path_func), exist_ok=True)
                    os.system(
                        f'mri_vol2vol --mov {ref_func} --targ {mask_path_anat} --lta {xform_func_to_anat} --inv --o {mask_path_func} --nearest')


            # ROI plots
            plot_dir = f'derivatives/masks/plots'
            os.makedirs(plot_dir, exist_ok=True)
            for space, ref, mask_path in zip(['anat', 'func'],
                                        [ref_anat, ref_func],
                                        [mask_path_anat, mask_path_func]):

                ref_range = os.popen(f'fslstats {ref} -R')
                ref_max = float(ref_range.read().split()[1])
                coords = os.popen(f'fslstats {mask_path} -C').read()[:-2].split(' ')
                coords = [int(float(c)) for c in coords]
                plot_file = f'{plot_dir}/sub-{subject}_{mask_label}_{space}.png'
                if not op.isfile(plot_file):
                    cmd = f'fsleyes render --outfile {plot_file} --size 3200 600 --scene ortho --autoDisplay -vl {coords[0]} {coords[1]} {coords[2]} ' \
                                     f'{ref} -dr 0 {ref_max} {mask_path} -dr 0 1 -cm greyscale'
                    os.system(cmd)

        # make links to reference images
        local_anat = f'{mask_dir}/anat_space/ref_anat.nii.gz'
        if not op.exists(local_anat):
            os.system(f'ln -s {ref_anat} {local_anat}')

        # make links to reference images
        local_func = f'{mask_dir}/func_space/ref_func.nii.gz'
        if not op.exists(local_func):
            os.system(f'ln -s {op.abspath(ref_func)} {local_func}')


# restrict roi mask to top n voxels above threshold in localizer contrast
n_vox_target = 512  # number of voxels to select from each ROI
thr_nvox = np.argsort(roi_data * loc_data)[n_vox_target]  # find lowest z value for top n_voxels
thr_final = max(thr, thr_nvox)  # use top n_vox or all vox above threshold, whichever is fewer
mask = np.array( (roi_data * loc_data) >= thr_final, dtype=int)
n_voxels = np.count_nonzero(mask)


# fake the registration so higher-level FEAT analyses can be run in func space
    # https://mumfordbrainstats.tumblr.com/post/166054797696/feat-registration-workaround
    reg_dirs = sorted(glob.glob(f"{os.getcwd()}/derivatives/FEAT/**/*.feat/reg", recursive=True))
    for reg_dir in reg_dirs:

        # delete all .mat files and replace with identity matrix
        for mat_path in glob.glob(f'{reg_dir}/*standard*'):
            os.remove(mat_path)
        shutil.copy(f'{os.environ["FSLDIR"]}/etc/flirtsch/ident.mat', f'{reg_dir}/example_func2standard.mat')

        # overwrite standard.nii.gz image with the mean_func.nii.gz
        shutil.copy(f'{op.dirname(reg_dir)}/mean_func.nii.gz', f'{reg_dir}/standard.nii.gz')


# plot (subject * level for each region)
            for region in fMRI.regions:
                df = df_analysis[(df_analysis.region == region)].copy()
                df['level'] = df['level'].astype('category').cat.reorder_categories(params['conds'])
                df_means = df.pivot(index='subject', columns='level', values='similarity')
                df_plot = df_means.plot(kind='bar',
                                        ylabel=ylabel,
                                        rot=0,
                                        figsize=figsizes['fMRI_exp2_ind'],
                                        color=params['colours'],
                                        legend=False)
                fig = df_plot.get_figure()
                plt.tick_params(direction='in')
                plt.xticks(rotation=25, ha="right", rotation_mode="anchor")
                plt.ylim(ylims)
                plt.title(f'region: {region}')
                plt.tight_layout()
                fig.savefig(f'{out_dir}/{region}_individual.png')
                plt.show()


"""
    # combine RSMs for exp1 and exp2
    if fMRI_exp == 'exp1+2':

        exp1_RSMs = pkl.load(open(f'{FMRI_DIR}/exp1/derivatives/RSA/' \
                         f'{analysis_path}/RSA.pkl', 'rb'))
        exp2_RSMs = pkl.load(open(f'{FMRI_DIR}/exp2/derivatives/RSA/' \
                                  f'{analysis_path}/RSA.pkl', 'rb'))
        RSMs_fMRI_matched = {}
        '''
        for subject in exp1_RSMs[layer]:
            RSMs_fMRI_matched[subject] = exp1_RSMs[layer][subject]
        for subject in exp2_RSMs[layer]:
            exp2_task_on = exp2_RSMs[layer][subject][:, :n_conds, :n_conds]
            if subject in RSMs_fMRI_matched:
                RSMs_fMRI_matched[subject] = np.concatenate(
                    [RSMs_fMRI_matched[subject], exp2_task_on], axis=0)
            else:
                RSMs_fMRI_matched[subject] = exp2_task_on
        '''
        for subject in exp2_RSMs[layer]:
            exp2_task_on = exp2_RSMs[layer][subject][:, :fMRI.n_img,
                           :fMRI.n_img]
            RSMs_fMRI_matched[subject] = exp2_task_on
        for subject in exp1_RSMs[layer]:
            if subject not in RSMs_fMRI_matched:
                RSMs_fMRI_matched[subject] = exp1_RSMs[layer][
                    subject]
    else:
    """

# load localizer contrast map (converting to native func space if necessary)
loc_path_std = f'{subj_dir}/task-{task}.gfeat/cope1.feat/stats' \
               f'/zstat1.nii.gz'
loc_path_anat = f'{loc_path_std[:-7]}_anat.nii.gz'
loc_path_func = f'{loc_path_std[:-7]}_func.nii.gz'

if not op.isfile(loc_path_func):

    reg_dir = f'{derdir}/ROIs/sub-{subject}/transforms'

    # std space to anat space
    ref_anat = f'{derdir}/ROIs/sub-{subject}/anat_space/ref_anat.nii.gz'
    xform_std_to_anat = f'{derdir}/sub-{subject}mri/transforms/' \
                        f'reg.mni152.2mm.lta'
    os.system(
        f'mri_vol2vol --mov {loc_path_std} --targ {ref_anat} --lta'
        f' {xform_std_to_anat} --o {loc_path_anat} --nearest')

    # anat space to func space
    ref_func = f'{derdir}/ROIs/sub-{subject}/func_space/ref_func.nii.gz'
    method = 'freesurfer' if subject == 'M012' and op.basename(
        os.getcwd()) == 'exp1' else 'FSL'
    if method == 'FSL':
        xform_anat_to_func = f'{reg_dir}/anat_to_func.mat'
        os.system(
            f'flirt -in {loc_path_anat} -ref {ref_func} -applyxfm '
            f'-init {xform_anat_to_func} -out {loc_path_func} '
            f'-interp nearestneighbour')

    elif method == 'freesurfer':
        xform_func_to_anat = f'{reg_dir}/func_to_anat.lta'
        os.system(
            f'mri_vol2vol --mov {ref_func} --targ {loc_path_anat} '
            f'--lta {xform_func_to_anat} --inv --o {loc_path_func}'
            f' --trilin')

    loc_data = nib.load(loc_path_func).get_fdata().flatten()


#Exp 2 norms

# use stats for each attention cond (exp 2 only)
elif norm == 'attention':
for attn in range(n_attn):
    run_idcs = np.arange(n_splits // 2) + (attn * n_splits // 2)
    norm_data[op][run_idcs] = np.tile(
        np_func(responses[run_idcs], axis=2,
                keepdims=True), (1, 1, n_img, 1))

# use stats for each attention * occluder cond (exp 2)
elif norm == 'attention-occluder':
    for attn, occ in itertools.product(range(n_attn),
                                       range(n_occ)):
        run_idcs = np.arange(n_splits // 2) + (attn * n_splits // 2)
        cond_idcs = np.arange(occ, n_img, n_occ)
        norm_data[op][run_idcs, :, :, :][:, :, cond_idcs, :] = \
            np.tile(np_func(responses[run_idcs, :, :, :][:, :, cond_idcs, :],
                            axis=2, keepdims=True), (1, 1, n_exem, 1))

# use stats for each attention cond, unoccluded only (exp 2)
elif norm == 'attention-unoccluded':
    for attn in range(n_attn):
        run_idcs = np.arange(n_splits // 2) + (attn * n_splits // 2)
        cond_idcs = np.arange(0, n_img, n_occ)
        norm_data[op][run_idcs, :, :, :] = np.tile(
            np_func(responses[run_idcs, :, :, :][:, :, cond_idcs, :],
                    axis=2, keepdims=True), (1, 1, n_img, 1))