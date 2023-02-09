'''
Implementation: Pranav Mani, Manley Roberts
'''

import torch
import numpy as np
import pickle
import pandas as pd
import os

from .experiment_utils import *

class DomainClassPriorMatrix:
    
    def get_class_priors(self):
        pass

    def get_hyperparameter_dict(self):
        pass

class PremadeClassPriorMatrix(DomainClassPriorMatrix):
    
    def __init__(self, n_classes, n_domains, assignment_matrix, min_train_num, min_test_num, min_valid_num):

        self.n_classes = n_classes
        self.n_domains = n_domains

        self.class_priors = assignment_matrix
        self.condition_number = np.linalg.cond(assignment_matrix)
        self.domain_relative_sizes = (np.ones(n_domains) / n_domains)

        fraction_needed_class = (self.domain_relative_sizes.T @ self.class_priors)
        max_fraction_needed = max(fraction_needed_class)

        assignment_scaler_train = min_train_num / max_fraction_needed
        assignment_scaler_test  = min_test_num / max_fraction_needed
        assignment_scaler_valid = min_valid_num / max_fraction_needed

        self.class_domain_assignment_matrix_train = torch.Tensor(np.floor(self.class_priors * self.domain_relative_sizes.reshape(n_domains, 1) * assignment_scaler_train))
        self.class_domain_assignment_matrix_test  = torch.Tensor(np.floor(self.class_priors * self.domain_relative_sizes.reshape(n_domains, 1) * assignment_scaler_test))
        self.class_domain_assignment_matrix_valid = torch.Tensor(np.floor(self.class_priors * self.domain_relative_sizes.reshape(n_domains, 1) * assignment_scaler_valid))

    def get_hyperparameter_dict(self):
        return {
            'name': get_name(self),
            'n_classes': self.n_classes,
            'n_domains': self.n_domains,
            'condition_number': self.condition_number,
            'matrices': {
                'class_prior': self.class_priors,
                'class_domain_assignment_matrix_train': self.class_domain_assignment_matrix_train,
                'class_domain_assignment_matrix_test': self.class_domain_assignment_matrix_test,
                'class_domain_assignment_matrix_valid': self.class_domain_assignment_matrix_valid
            },
            'n_samples_train': torch.sum(self.class_domain_assignment_matrix_train),
            'n_samples_test': torch.sum(self.class_domain_assignment_matrix_test),
            'n_samples_valid': torch.sum(self.class_domain_assignment_matrix_valid),
        }

class RandomDomainClassPriorMatrix(DomainClassPriorMatrix):

    def __init__(self, n_classes, n_domains, max_condition_number, random_seed, class_prior_alpha, min_train_num, min_test_num, min_valid_num, save_folder, template_file):

        self.n_classes = n_classes
        self.n_domains = n_domains
        self.random_seed = random_seed

        self.class_prior_alpha = class_prior_alpha

        self.save_folder = save_folder

        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        save_local = f'{self.n_classes}_{self.n_domains}_{self.class_prior_alpha}_{self.random_seed}'
        path_ending = '.pickle'
        self.matrix_load_path = save_folder + save_local + path_ending

        if os.path.exists(self.matrix_load_path):
            with open(self.matrix_load_path, 'rb') as h:
                print(f'Loaded {self.matrix_load_path}')
                load_dict = pickle.load(h)
                self.class_priors, self.condition_number = load_dict['class_priors'], load_dict['condition_number']
        else:
            if template_file is not None and os.path.exists(template_file):
                dataframe_lookup = pd.read_csv(template_file)
                output_matrix, filter_matrix, known_cond = self.get_class_prior_template(n_classes, n_domains, class_prior_alpha, random_seed, dataframe_lookup)
                template = (
                    output_matrix, filter_matrix, known_cond
                )
            else:
                template = None
            self.class_priors, self.condition_number = self.generate_class_priors(n_classes, n_domains, max_condition_number, random_seed, class_prior_alpha, template=template)
            save_dict = {
                'class_priors': self.class_priors,
                'condition_number': self.condition_number
            }
            with open(self.matrix_load_path, 'wb') as h:
                pickle.dump(save_dict, h, protocol=pickle.HIGHEST_PROTOCOL)

        self.domain_relative_sizes = (np.ones(n_domains) / n_domains)

        fraction_needed_class = (self.domain_relative_sizes.T @ self.class_priors)
        max_fraction_needed = max(fraction_needed_class)

        assignment_scaler_train = min_train_num / max_fraction_needed
        assignment_scaler_test  = min_test_num / max_fraction_needed
        assignment_scaler_valid = min_valid_num / max_fraction_needed

        self.class_domain_assignment_matrix_train = torch.Tensor(np.floor(self.class_priors * self.domain_relative_sizes.reshape(n_domains, 1) * assignment_scaler_train))
        self.class_domain_assignment_matrix_test  = torch.Tensor(np.floor(self.class_priors * self.domain_relative_sizes.reshape(n_domains, 1) * assignment_scaler_test))
        self.class_domain_assignment_matrix_valid = torch.Tensor(np.floor(self.class_priors * self.domain_relative_sizes.reshape(n_domains, 1) * assignment_scaler_valid))

    def get_class_prior_template(self, n_classes, n_domains, alpha, seed, dataframe_lookup):

        filtered_data = dataframe_lookup[
            (dataframe_lookup['class_prior.n_classes'] == n_classes) & 
            (dataframe_lookup['class_prior.n_domains'] == n_domains) & 
            (dataframe_lookup['class_prior.alpha'] == alpha) &
            (dataframe_lookup['class_prior.random_seed'] == seed)
        ]

        # assert(len(filtered_data) == 1)
        lookup_entry = filtered_data.iloc[0]

        string_to_parse = lookup_entry['class_prior.matrices.class_prior']
        # num_classes     = lookup_entry['class_prior.n_classes']
        # num_domains     = lookup_entry['class_prior.n_domains']
        known_cond      = lookup_entry['class_prior.condition_number']
        rows = string_to_parse.replace('\n',' ').replace('[',' ').replace(',',' ').split(']')
        list_rows = []
        top_bottom_border = -1
        left_right_border = -1
        for i, row in enumerate(rows):
            print(row)
            entries = row.split()
            if len(entries) > 0:
                if entries[0] == '...':
                    top_bottom_border = i
                elif '...' in entries:
                    left_right_border = (len(entries) - 1) // 2

                num_list = np.array([float(e) for e in filter(lambda e: e != '...', entries)])
                list_rows.append(num_list)
        if top_bottom_border == -1 and left_right_border == -1:
            assert(len(list_rows) == n_domains)
            output_matrix = np.stack(list_rows, axis=0)
            filter_matrix = np.ones((n_domains, n_classes))
        else:
            border_matrix = np.stack(list_rows, axis=0)
            output_matrix = np.zeros((n_domains, n_classes))
            filter_matrix = np.zeros((n_domains, n_classes))
            filter_matrix[:top_bottom_border, :left_right_border] = 1
            filter_matrix[-top_bottom_border:, :left_right_border] = 1
            filter_matrix[:top_bottom_border, -left_right_border:] = 1
            filter_matrix[-top_bottom_border:, -left_right_border:] = 1

            output_matrix[:top_bottom_border, :left_right_border] = border_matrix[:top_bottom_border, :left_right_border]
            output_matrix[-top_bottom_border:, :left_right_border] = border_matrix[-top_bottom_border:, :left_right_border]         
            output_matrix[:top_bottom_border, -left_right_border:] = border_matrix[:top_bottom_border, -left_right_border:]
            output_matrix[-top_bottom_border:, -left_right_border:] = border_matrix[-top_bottom_border:, -left_right_border:] 
        return output_matrix, filter_matrix, known_cond

    def get_hyperparameter_dict(self):
        return {
            'name': get_name(self),
            'n_classes': self.n_classes,
            'n_domains': self.n_domains,
            'random_seed': self.random_seed,
            'condition_number': self.condition_number,
            'matrices': {
                'class_prior': self.class_priors,
                'class_domain_assignment_matrix_train': self.class_domain_assignment_matrix_train,
                'class_domain_assignment_matrix_test': self.class_domain_assignment_matrix_test,
                'class_domain_assignment_matrix_valid': self.class_domain_assignment_matrix_valid
            },
            'n_samples_train': torch.sum(self.class_domain_assignment_matrix_train),
            'n_samples_test': torch.sum(self.class_domain_assignment_matrix_test),
            'n_samples_valid': torch.sum(self.class_domain_assignment_matrix_valid),
            'alpha': self.class_prior_alpha
        }

    def generate_class_priors(self, n_classes, n_domains, max_condition_number, random_seed, class_prior_alpha, template=None):
        if random_seed is not None:
            np.random.seed(random_seed)
        condition = max_condition_number + 1
        class_prior_alpha_vec = class_prior_alpha * (np.ones(n_classes) / n_classes)

        if template is not None:
            expected, filter, known_cond = template

            if np.sum(filter - 1) == 0:
                return expected, known_cond

        generated = 0
        error = 0
        while condition > max_condition_number:

            class_priors = np.random.dirichlet(class_prior_alpha_vec, n_domains) 
            generated += 1

            if template is not None:
                error = np.sum(
                    filter * (
                        expected - class_priors
                    )
                )

            # only if this is actually a potential candidate
            if error < 1e-6:
                print(generated)
                condition = np.linalg.cond(class_priors)

        class_priors_adjusted = np.where(class_priors < 1e-7, 0, class_priors)
        class_priors_adjusted = class_priors_adjusted / np.sum(class_priors_adjusted, axis=1, keepdims=True)

        class_priors = class_priors_adjusted

        return class_priors, condition