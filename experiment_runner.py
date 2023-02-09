import os

import argparse
import yaml
import wandb
import pickle
import pathlib

from datetime import datetime

parser = argparse.ArgumentParser(description='Usually, the necessary argument is the path to a yml file describing experiments. If certain flags are placed, the experiments to be conducted will be replaced by certain fixed paper replication suites.')
parser.add_argument('--dataset_config_path', type=str, default='experiment_config.yml',
                    help='The path to the experiment config yaml file')
parser.add_argument('--replace_experiments_with_naive_ablation',type=str,default=None,
                    help='Replace experiments in config yaml file with ablation carried out using naive scan representations for a specific dataset')
parser.add_argument('--replace_experiments_with_paper_main', type=str, default=None,
                    help='Replace experiments in config yaml file with the paper main experiments for a specific dataset')
parser.add_argument('--replace_experiments_with_paper_cluster_ablation',default=False,action=argparse.BooleanOptionalAction,
                    help='Replace experiments in config yaml file with naive ablation for CIFAR20, 20 domains, DDFA (SI) only.')
parser.add_argument('--GPU', type=int, default=-1,
                    help='GPU to use (-1 for CPU)')
parser.add_argument('--start_late', type=int, default=0, help='Paper replication experiments are executed in order. This will skip to a certain index. Helpful if a run is interrupted.')
parser.add_argument('--num_seeds', type=int, default=5, help='Number of the specified random seeds to use when replicating paper experiments.')
args = parser.parse_args()

with open(args.dataset_config_path) as f:
    experiment_config = yaml.full_load(f)

if args.GPU != -1:
    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.GPU)    
    import torch
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
else:
    import torch
    device = 'cpu'

print('Using device: ', device)

from ddfa.components.dataset import *
from ddfa.components.permutation_solver import *
from ddfa.components.experiment_utils import *
from ddfa.components.domain_discriminator.scan_model_definitions import *
from ddfa.components.domain_discriminator.domain_discriminator_scan import * 
from ddfa.components.experiment_framework import *

# FOR CONFIGURING PAPER MAIN EXPERIMENTS:
# The following code will replace the experiment_config['experiments'] attributes with those for replicating paper results

if args.replace_experiments_with_naive_ablation is not None: 
    
    experiment_config['experiments'] = [] 
    dataset_choice = args.replace_experiments_with_naive_ablation
    dataset_root = experiment_config['datasets'][dataset_choice]['root_path'] 

    paper_replication_experiments = experiment_config['paper_replication_experiments']

    for i, random_seed_wave in enumerate(paper_replication_experiments['random_seed_waves']):
        if i >= args.num_seeds:
            break
        for alpha, max_cond_number in \
                zip(
                    paper_replication_experiments['alphas'],
                    paper_replication_experiments['datasets'][dataset_choice]['max_condition_numbers']
                ):
            # for domains, data_generation_seed, model_stochasticity_seed in \
            for domains, data_generation_seed in \
                zip(
                    paper_replication_experiments['datasets'][dataset_choice]['domains'],
                    paper_replication_experiments['random_seed_waves'][random_seed_wave]['data_generation_seed'],
                    # paper_replication_experiments['random_seed_waves'][random_seed_wave]['model_stochasticity_seed'],   
                ):

                for approach in ['naive_scan','naive_pca','naive_ica']:
                    run_dict = {
                        'dataset_settings': {
                            'dataset': dataset_choice
                        },
                        'class_prior_generation': {
                            'domains': domains,
                            'alpha': alpha,
                            'max_condition_number': max_cond_number,
                        },
                        'class_prior_estimator': 'cluster_nmf',
                        'data_generation_seed': data_generation_seed,
                        # 'model_stochasticity_seed': model_stochasticity_seed,
                        'estimate_prior_valid_train': True,
                        'retrain': False,
                        'approaches': approach
                    }
                    experiment_config['experiments'].append(run_dict) 
        
elif args.replace_experiments_with_paper_cluster_ablation:    
    experiment_config['experiments'] = []

    dataset_choice = 'cifar20'
    dataset_root    = experiment_config['datasets'][dataset_choice]['root_path']

    n_classes       = experiment_config['datasets'][dataset_choice]['classes']

    paper_replication_experiments = experiment_config['paper_replication_experiments']

    for i, random_seed_wave in enumerate(paper_replication_experiments['random_seed_waves']):
        if i >= args.num_seeds:
            break
        for alpha, max_cond_number in \
                zip(
                    paper_replication_experiments['alphas'],
                    paper_replication_experiments['datasets'][dataset_choice]['max_condition_numbers']
                ):
            # for domains, data_generation_seed, model_stochasticity_seed in \
            for domains, data_generation_seed in \
                zip(
                    paper_replication_experiments['datasets'][dataset_choice]['domains'],
                    paper_replication_experiments['random_seed_waves'][random_seed_wave]['data_generation_seed'],
                    # paper_replication_experiments['random_seed_waves'][random_seed_wave]['model_stochasticity_seed'],   
                ):
                if domains == n_classes:
                    run_dict = {
                        'dataset_settings': {
                            'dataset': dataset_choice
                        },
                        'class_prior_generation': {
                            'domains': domains,
                            'alpha': alpha,
                            'max_condition_number': max_cond_number,
                        },                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       

                        'class_prior_estimator': 'cluster_nmf_ablation',
                        'data_generation_seed': data_generation_seed,
                        # 'model_stochasticity_seed': model_stochasticity_seed,
                        'estimate_prior_valid_train': True,
                        'retrain': False,
                        'approaches': ['ddfa_scan']
                    }
                    experiment_config['experiments'].append(run_dict)


elif args.replace_experiments_with_paper_main is not None:
    experiment_config['experiments'] = []

    dataset_choice = args.replace_experiments_with_paper_main
    dataset_root    = experiment_config['datasets'][dataset_choice]['root_path']

    paper_replication_experiments = experiment_config['paper_replication_experiments']

    for i, random_seed_wave in enumerate(paper_replication_experiments['random_seed_waves']):
        if i >= args.num_seeds:
            break
        for alpha, max_cond_number in \
                zip(
                    paper_replication_experiments['alphas'],
                    paper_replication_experiments['datasets'][dataset_choice]['max_condition_numbers']
                ):
            # for domains, data_generation_seed, model_stochasticity_seed in \
            for domains, data_generation_seed in \
                zip(
                    paper_replication_experiments['datasets'][dataset_choice]['domains'],
                    paper_replication_experiments['random_seed_waves'][random_seed_wave]['data_generation_seed'],
                    # paper_replication_experiments['random_seed_waves'][random_seed_wave]['model_stochasticity_seed'],   
                ):
                run_dict = {
                    'dataset_settings': {
                        'dataset': dataset_choice
                    },
                    'class_prior_generation': {
                        'domains': domains,
                        'alpha': alpha,
                        'max_condition_number': max_cond_number,
                    },
                    'class_prior_estimator': 'cluster_nmf',
                    'data_generation_seed': data_generation_seed,
                    # 'model_stochasticity_seed': model_stochasticity_seed,
                    'estimate_prior_valid_train': True,
                    'retrain': False,
                    'approaches': ['ddfa_scan']
                }
                if dataset_choice in ['cifar10', 'cifar20']:
                    run_dict['approaches'].append('ddfa')
                experiment_config['experiments'].append(run_dict)

use_wandb           = experiment_config['wandb']['use_wandb']
if use_wandb:
    wandb_entity    = experiment_config['wandb']['entity']
    wandb_project   = experiment_config['wandb']['project']

dummy_dataset_instance = None

for experiment in experiment_config['experiments'][args.start_late:]:
    start_time = datetime.now()

    dataset_choice  = experiment['dataset_settings']['dataset']

    dataset_root    = experiment_config['datasets'][dataset_choice]['root_path']

    domains         = experiment['class_prior_generation']['domains']
    alpha           = experiment['class_prior_generation']['alpha']
    max_cond_number = experiment['class_prior_generation']['max_condition_number']

    data_generation_seed = experiment['data_generation_seed']
    np.random.seed(data_generation_seed)
    class_prior_seed = np.random.randint(1, 4_000_000_000)
    dataset_seed     = np.random.randint(1, 4_000_000_000)
    assignment_seed  = np.random.randint(1, 4_000_000_000)

    # originally intended to seed here with model_stochasticity_seed. 
    # Using the same seed to seed two different objects is odd but should not cause problems as we are seeding very different things (e.g. random dirichlet sampling and then random weight initialization).
    # this also means model_stochasticity_seed is currently unused.
    np.random.seed(data_generation_seed)
    dd_stochasticity_seed       = np.random.randint(1, 4_000_000_000)
    class_prior_estimation_seed = np.random.randint(1, 4_000_000_000)
    kmeans_seed                 = np.random.randint(1, 4_000_000_000)

    use_raw_ddfa       = 'ddfa'          in experiment['approaches']
    use_scan           = 'ddfa_scan'     in experiment['approaches']
    use_naive_scan     = 'naive_scan'    in experiment['approaches'] 
    use_naive_pca      = 'naive_pca'     in experiment['approaches'] 
    use_naive_ica      = 'naive_ica'     in experiment['approaches'] 

    estimate_prior_valid_train = experiment['estimate_prior_valid_train']

    retrain         = experiment['retrain']

    class_prior_estimator_choice = experiment['class_prior_estimator']
    lambda_threshold             = experiment['lambda_threshold'] if 'lambda_threshold' in experiment else None

    print(f'Setup Elapsed: {datetime.now() - start_time}, finished loading params')

    if dataset_choice == 'cifar10':
        if dummy_dataset_instance is None or not isinstance(dummy_dataset_instance, CIFAR10):
            dummy_dataset_instance = CIFAR10(data_root=dataset_root, batch_size=32, dataset_seed=42)
        dataset_class = CIFAR10

        scan_ddfa_epochs        = 25
        scan_ddfa_loadpath      = './pretrain/scan_cifar_pretrain/scan_cifar-10.pth.tar'
        scan_ddfa_subclass_name = scan_scan
        baseline_scan_name      = scan_ddfa_loadpath

        if use_naive_scan: 
            scan_naive_subclass_name = scan_scan_naive
        elif use_naive_pca:
            scan_naive_subclass_name = scan_scan_naive_pca
        elif use_naive_ica:
            scan_naive_subclass_name = scan_scan_naive_ica 

        ddfa_epochs             = 100
        ddfa_n_discretization   = 30

    elif dataset_choice == 'cifar20':
        if dummy_dataset_instance is None or not isinstance(dummy_dataset_instance, CIFAR20):
            dummy_dataset_instance = CIFAR20(data_root=dataset_root, batch_size=32, dataset_seed=42)
        dataset_class = CIFAR20

        ablation_cluster_numbers = [10, 35, 50, 100, 150] # exclude 20, it is handled elsewhere in main paper experiments

        scan_ddfa_epochs        = 25
        scan_ddfa_loadpath      = './pretrain/scan_cifar_pretrain/scan_cifar-20.pth.tar'
        scan_ddfa_subclass_name = scan_scan

        baseline_scan_name      = scan_ddfa_loadpath

        if use_naive_scan: 
            scan_naive_subclass_name = scan_scan_naive
        elif use_naive_pca:
            scan_naive_subclass_name = scan_scan_naive_pca
        elif use_naive_ica:
            scan_naive_subclass_name = scan_scan_naive_ica 

        ddfa_epochs             = 100
        ddfa_n_discretization   = 60

    elif dataset_choice == 'imagenet':
        if dummy_dataset_instance is None or not isinstance(dummy_dataset_instance, ImageNet50):
            dummy_dataset_instance = ImageNet50(data_root=dataset_root, batch_size=32, dataset_seed=42)
        dataset_class = ImageNet50

        scan_ddfa_epochs        = 25
        scan_ddfa_loadpath      = './pretrain/scan_imagenet_pretrain/scan_imagenet_50.pth.tar'
        scan_ddfa_subclass_name = scan_scan_imagenet

        baseline_scan_name      = scan_ddfa_loadpath

        if use_naive_scan: 
            scan_naive_subclass_name = scan_scan_naive
        elif use_naive_pca:
            scan_naive_subclass_name = scan_scan_naive_pca
        elif use_naive_ica:
            scan_naive_subclass_name = scan_scan_naive_ica 

    elif dataset_choice == 'fg2':
        if dummy_dataset_instance is None or not isinstance(dummy_dataset_instance, FieldGuide2):
            dummy_dataset_instance = FieldGuide2(data_root=dataset_root, batch_size=32, dataset_seed=42)
        dataset_class = FieldGuide2

        scan_ddfa_epochs        = 30
        scan_ddfa_loadpath      = './pretrain/scan_fieldguide_pretrain/fieldguide2/pretext/model.pth.tar'
        scan_ddfa_subclass_name = scan_pretext

        # for comparison
        baseline_scan_name      = './pretrain/scan_fieldguide_pretrain/fieldguide2/scan/model.pth.tar'

        if use_naive_scan: 
            scan_naive_subclass_name = scan_scan_naive
        elif use_naive_pca:
            scan_naive_subclass_name = scan_scan_naive_pca
        elif use_naive_ica:
            scan_naive_subclass_name = scan_scan_naive_ica 

    elif dataset_choice == 'fg28':
        if dummy_dataset_instance is None or not isinstance(dummy_dataset_instance, FieldGuide28):
            dummy_dataset_instance = FieldGuide28(data_root=dataset_root, batch_size=32, dataset_seed=42)
        dataset_class = FieldGuide28

        scan_ddfa_epochs        = 60
        scan_ddfa_loadpath      = './pretrain/scan_fieldguide_pretrain/fieldguide28/pretext/model.pth.tar'
        scan_ddfa_subclass_name = scan_pretext
        # for comparison
        baseline_scan_name      = './pretrain/scan_fieldguide_pretrain/fieldguide28/scan/model.pth.tar'

        if use_naive_scan: 
            scan_naive_subclass_name = scan_scan_naive
        elif use_naive_pca:
            scan_naive_subclass_name = scan_scan_naive_pca
        elif use_naive_ica:
            scan_naive_subclass_name = scan_scan_naive_ica 

    runs = []

    print(f'Setup Elapsed: {datetime.now() - start_time}, started loading data')

    dataset_instance = dataset_class(data_root=dataset_root, batch_size=32, dataset_seed=dataset_seed)

    print(f'Setup Elapsed: {datetime.now() - start_time}, finished loading data')

    class_prior = RandomDomainClassPriorMatrix(
        n_classes = dummy_dataset_instance.n_classes, 
        n_domains = domains, 
        max_condition_number = max_cond_number, 
        random_seed = class_prior_seed, 
        class_prior_alpha = alpha, 
        min_train_num = dummy_dataset_instance.min_train_num,
        min_test_num = dummy_dataset_instance.min_test_num, 
        min_valid_num = dummy_dataset_instance.min_valid_num,
        save_folder = experiment_config['save_path_generated_matrices'],
        template_file = experiment_config['template_file'],
    )

    print(dataset_instance.min_train_num, dummy_dataset_instance.min_train_num)
    print(dataset_instance.min_test_num, dummy_dataset_instance.min_test_num)
    print(dataset_instance.min_valid_num, dummy_dataset_instance.min_valid_num)

    print(f'Setup Elapsed: {datetime.now() - start_time}, finished generating class prior matrix')

    if use_scan:
        # Add scan main run 

        path_root = './experiments/'
        discriminator_name = get_name_class(scan_ddfa_subclass_name)
        path_specs = f'{dataset_choice}_{domains}_{alpha}_{max_cond_number}_{discriminator_name}_{data_generation_seed}'
        # path_specs = f'{dataset_choice}_{domains}_{alpha}_{max_cond_number}_{discriminator_name}_{data_generation_seed}_{model_stochasticity_seed}'
        path_ending = '.pt'
        weight_load_path = path_root + path_specs + path_ending

        start_time = datetime.now()
        if class_prior_estimator_choice == 'cluster_nmf_ablation':
            class_prior_estimators = [ClusterNMFClassPriorEstimation(
                    base_cluster_model = ClusterModelFaissKMeans(use_gpu=True, random_seed=kmeans_seed),
                    n_discretization = m,
                    class_prior_estimation_seed = class_prior_estimation_seed
                ) for m in ablation_cluster_numbers
            ]

            for class_prior_estimator in class_prior_estimators:
                runs.append({
                'n_domains': domains,
                'class_prior': class_prior,
                'class_prior_estimator': class_prior_estimator,
                'dataset': dataset_instance,
                'permutation_solver': ScipyOptimizeLinearSumPermutationSolver(),
                'discriminator': scan_ddfa_subclass_name(
                        device,
                        dd_stochasticity_seed = dd_stochasticity_seed,
                        lr = 0.00001,
                        exp_lr_gamma = 0.97,
                        epochs = scan_ddfa_epochs,
                        batch_size = 32,
                        n_classes= class_prior.n_classes,
                        n_domains = domains,
                        load_path= scan_ddfa_loadpath,                    
                        eval_ps = ScipyOptimizeLinearSumPermutationSolver(),
                        class_prior = class_prior,
                        dropout = 0,
                        limit_gradient_flow=False,
                        use_scheduler = 'ExponentialLR',
                        baseline_load_path=baseline_scan_name,
                        weight_load_path = None if retrain else weight_load_path,
                        use_wandb = use_wandb
                ),
                'alpha': alpha,
                'data_generation_seed': data_generation_seed,
                # 'model_stochasticity_seed': model_stochasticity_seed,
                'estimate_prior_valid_train': estimate_prior_valid_train,
                'naive':False,
                })

        else:
            if class_prior_estimator_choice == 'cluster_nmf':
                class_prior_estimator = ClusterNMFClassPriorEstimation(
                        base_cluster_model = ClusterModelFaissKMeans(use_gpu=True, random_seed=kmeans_seed),
                        n_discretization = dummy_dataset_instance.n_classes,
                        class_prior_estimation_seed = class_prior_estimation_seed
                )
            else:
                class_prior_estimator = None
            runs.append({
                'n_domains': domains,
                'class_prior': class_prior,
                'class_prior_estimator': class_prior_estimator,
                'dataset': dataset_instance,
                'permutation_solver': ScipyOptimizeLinearSumPermutationSolver(),
                'discriminator': scan_ddfa_subclass_name(
                        device,
                        dd_stochasticity_seed = dd_stochasticity_seed,
                        lr = 0.00001,
                        exp_lr_gamma = 0.97,
                        epochs = scan_ddfa_epochs,
                        batch_size = 32,
                        n_classes= class_prior.n_classes,
                        n_domains = domains,
                        load_path= scan_ddfa_loadpath,                    
                        eval_ps = ScipyOptimizeLinearSumPermutationSolver(),
                        class_prior = class_prior,
                        dropout = 0,
                        limit_gradient_flow=False,
                        use_scheduler = 'ExponentialLR',
                        baseline_load_path=baseline_scan_name,
                        weight_load_path = None if retrain else weight_load_path,
                        use_wandb = use_wandb
                ),
                'alpha': alpha,
                'data_generation_seed': data_generation_seed,
                # 'model_stochasticity_seed': model_stochasticity_seed,
                'estimate_prior_valid_train': estimate_prior_valid_train,
                'naive':False,
            })


    if use_raw_ddfa:      

        path_root = './experiments/'
        discriminator_name = 'CIFAR10PytorchCifar'
        path_specs = f'{dataset_choice}_{domains}_{alpha}_{max_cond_number}_{discriminator_name}_{data_generation_seed}'
        # path_specs = f'{dataset_choice}_{domains}_{alpha}_{max_cond_number}_{discriminator_name}_{data_generation_seed}_{model_stochasticity_seed}'
        path_ending = '.pt'
        weight_load_path = path_root + path_specs + path_ending

        # Add Domain Discriminator run
        if class_prior_estimator_choice == 'cluster_nmf':
            class_prior_estimator = ClusterNMFClassPriorEstimation(
                base_cluster_model = ClusterModelFaissKMeans(use_gpu=True, random_seed=kmeans_seed),
                n_discretization = ddfa_n_discretization,
                class_prior_estimation_seed = class_prior_estimation_seed
            )
        else:
            class_prior_estimator = None
        runs.append({
            'n_domains': domains,
            'class_prior': class_prior,
            'class_prior_estimator': class_prior_estimator,
            'dataset': dataset_instance,
            'permutation_solver': ScipyOptimizeLinearSumPermutationSolver(),
            'discriminator': CIFAR10PytorchCifar(
            # 'extractor': CIFAR10PytorchCifar(
                device = device,
                dd_stochasticity_seed = dd_stochasticity_seed,
                lr = 0.001,
                exp_lr_gamma = 0.97,

                epochs = ddfa_epochs,

                batch_size = 32,
                n_classes = class_prior.n_classes,
                n_domains = domains,
                eval_ps = ScipyOptimizeLinearSumPermutationSolver(),
                class_prior = class_prior,
                weight_load_path = None if retrain else weight_load_path,
                use_wandb = use_wandb
            ),
            'alpha': alpha,
            'data_generation_seed': data_generation_seed,
            # 'model_stochasticity_seed': model_stochasticity_seed,
            'estimate_prior_valid_train': estimate_prior_valid_train,
            'naive':False,
        })
        
    if use_naive_scan or use_naive_ica or use_naive_pca: 
        
        if dataset_choice in ['cifar10','cifar20']:
            architecture = 'scan_scan'
        elif dataset_choice in ['imagenet']:
            architecture = 'scan_imagenet' 
        else:
            architecture = 'scan_pretext' 

        path_root = './experiments/'
        discriminator_name = get_name_class(scan_naive_subclass_name)
        # path_specs = f'{dataset_choice}_{domains}_{alpha}_{max_cond_number}_{discriminator_name}_{data_generation_seed}_{model_stochasticity_seed}'
        path_specs = f'{dataset_choice}_{domains}_{alpha}_{max_cond_number}_{discriminator_name}_{data_generation_seed}'
        path_ending = '.pt'
        weight_load_path = path_root + path_specs + path_ending

        if class_prior_estimator_choice == 'cluster_nmf':
            class_prior_estimator = ClusterNMFClassPriorEstimation(
                    base_cluster_model = ClusterModelFaissKMeans(use_gpu=True, random_seed=kmeans_seed),
                    n_discretization = dummy_dataset_instance.n_classes,
                    class_prior_estimation_seed = class_prior_estimation_seed
                )
        else:
            class_prior_estimator = None
        runs.append({
            'n_domains': domains,
            'class_prior': class_prior,
            'class_prior_estimator': class_prior_estimator,
            'dataset': dataset_instance,
            'permutation_solver': ScipyOptimizeLinearSumPermutationSolver(),
            'discriminator': scan_naive_subclass_name(
                    device,
                    n_domains = domains,
                    n_classes = class_prior.n_classes,
                    architecture = architecture,
                    dd_stochasticity_seed = dd_stochasticity_seed,
                    eval_ps = ScipyOptimizeLinearSumPermutationSolver(),
                    load_path=scan_ddfa_loadpath,
                    baseline_load_path=baseline_scan_name),
                    
            'alpha': alpha,
            'data_generation_seed': data_generation_seed,
            # 'model_stochasticity_seed': model_stochasticity_seed,
            'estimate_prior_valid_train': estimate_prior_valid_train,
            'naive':True,
        })


    for r in runs:

        n_domains               = r['n_domains']
        class_prior             = r['class_prior']
        class_prior_estimator   = r['class_prior_estimator']
        permutation_solver      = r['permutation_solver']
        discriminator           = r['discriminator']
        dataset_instance        = r['dataset']
        estimate_prior_valid_train = r['estimate_prior_valid_train']

        naive = r['naive']

        config = {
            component_name : None if component is None else component.get_hyperparameter_dict()
            for component_name, component in [
                ('dataset', dataset_instance),
                ('class_prior', class_prior),
                ('class_prior_estimator', class_prior_estimator),
                ('permutation_solver', permutation_solver),
                ('discriminator', discriminator)
            ]
        }

        config.update({
            'data_generation_seed': data_generation_seed,
            # 'model_stochasticity_seed': model_stochasticity_seed,
            'estimate_prior_valid_train': estimate_prior_valid_train,
        })

        if use_wandb:
            run = wandb.init(
                entity=wandb_entity,
                project=wandb_project,
                reinit=True,
                config=config
            )

        experiment = ExperimentSetup(dataset_instance, class_prior, discriminator, class_prior_estimator, permutation_solver, device, assignment_seed, batch_size=32, estimate_prior_valid_train=estimate_prior_valid_train, naive=naive)

        if class_prior_estimator is not None:
            result_dict = {
                # "final_best_labels": list(experiment.permuted_labels),
                'test_post_cluster_acc': experiment.test_post_cluster_acc,
                'test_post_cluster_p_y_given_d_l1_norm': experiment.test_post_cluster_p_y_given_d_l1_norm
            }
            if hasattr(experiment, 'scan_alone_test_acc'):
                result_dict.update({'scan_alone_best_acc': experiment.scan_alone_test_acc})
            if hasattr(experiment, 'scan_alone_reconstruction_error_L1'):
                result_dict.update({'scan_alone_reconstruction_error_L1': experiment.scan_alone_reconstruction_error_L1})
            if hasattr(experiment, 'scan_reconstructed_p_y_given_d'):
                result_dict.update({'scan_reconstructed_p_y_given_d': experiment.scan_reconstructed_p_y_given_d})
            
            if use_wandb:
                wandb.config.update(result_dict)
        else:
            result_dict = None


        run_summary_dict = {
            'config': config,
            'result_dict': result_dict,
        }

        # print(run_summary_dict)
        print(config['class_prior_estimator'], result_dict['test_post_cluster_acc'], result_dict['test_post_cluster_p_y_given_d_l1_norm'])

        # Save back model checkpoint for best model
        checkpoint_path_root = './experiments/'
        discriminator_name   = get_name(discriminator)
        # path_specs = f'{dataset_choice}_{domains}_{alpha}_{max_cond_number}_{discriminator_name}_{data_generation_seed}_{model_stochasticity_seed}'
        path_specs = f'{dataset_choice}_{domains}_{alpha}_{max_cond_number}_{discriminator_name}_{data_generation_seed}'
        checkpoint_path_ending = '.pt'
        checkpoint_save_path = checkpoint_path_root + path_specs + checkpoint_path_ending
        if not os.path.exists(checkpoint_path_root):
            os.makedirs(checkpoint_path_root)

        domain_discriminator_weights = experiment.domain_discriminator.model.cpu().state_dict()
        torch.save(domain_discriminator_weights, checkpoint_save_path)

        # Save back results
        results_path_root = './results/'
        results_path_ending = '.pickle'
        results_save_path = results_path_root + path_specs + results_path_ending
        if not os.path.exists(results_path_root):
            os.makedirs(results_path_root)

        with open(str(pathlib.Path(results_save_path).absolute()), 'wb') as h:
            pickle.dump(run_summary_dict, h, protocol=pickle.HIGHEST_PROTOCOL)

        if use_wandb:
            wandb.save(checkpoint_path_root)
            wandb.save(results_path_root)