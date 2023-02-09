'''
Implementation: Pranav Mani, Manley Roberts
'''

import numpy as np
import torch

def label_accuracy(label_output, true_classes):
    return 1 - np.count_nonzero(label_output - true_classes) / len(label_output)

def get_name(object):
    return object.__class__.__name__

def get_name_class(object):
    return object.__name__

def y_predictions_dd_uniform(p_d_x, p_y_d):
    p_y_x = solve_p_y_x(p_d_x, p_y_d)
    return np.argmax(p_y_x, axis=1), p_y_x

def y_predictions_dd_balanced(p_y_d, p_y_x, true_domains):
    p_y_x_d = solve_p_y_x_d(p_y_d, p_y_x, true_domains)
    return np.argmax(p_y_x_d, axis=1), p_y_x_d

def solve_p_y_x(p_d_x, p_y_d):
    # note p_d_x batched
    p_d_x_t = p_d_x.T
    p_d_y = p_y_d.T / np.where(np.sum(p_y_d.T, axis=0, keepdims=True) == 0, 1e-8, np.sum(p_y_d.T, axis=0, keepdims=True))
    p_y_x_t = np.linalg.pinv(p_d_y) @ p_d_x_t
    p_y_x = p_y_x_t.T
    return p_y_x

def solve_p_y_x_d(p_y_d, p_y_x, true_domains):

    p_d_y = p_y_d.T / np.where(np.sum(p_y_d.T, axis=0, keepdims=True) == 0, 1e-8, np.sum(p_y_d.T, axis=0, keepdims=True))

    p_d_y_list = [
        p_d_y[domain:domain+1, :]
        for domain in true_domains
    ]
    # each row is p(d_true | y)
    p_d_y_rows = np.concatenate(p_d_y_list, axis=0)

    p_y_x_d = p_y_x * p_d_y_rows
    p_y_x_d /= np.sum(p_y_x_d, axis=1, keepdims=True) # normalize rows to account for denominator
    return p_y_x_d

def model_evaluate(model, dataset_loader, input_data, input_domains, true_p_y_d, ps, device): 

    test_labels = []

    model.eval()
    output_list = []

    per_domain_class_counts = np.zeros_like(true_p_y_d)

    for vec in dataset_loader:
        if isinstance(vec, dict):
            batch, labels = vec['image'], vec['target']
        else:
            batch, labels = vec
        batch = batch.to(device)
        with torch.no_grad():
            cluster_softmax = model(batch)[0] 
            _ , cluster_assignments = torch.max(cluster_softmax,dim=1)
            output_list.append(cluster_assignments.cpu().numpy()) 
        test_labels.append(labels.cpu().numpy())

    input_predictions_list = []
    for input_dataset_loader in input_data:
        for vec in input_dataset_loader:
            if isinstance(vec, dict):
                batch, labels = vec['image'], vec['target']
            else:
                batch, labels = vec
            batch = batch.to(device)
            with torch.no_grad():
                cluster_softmax = model(batch)[0] 
                _ , cluster_assignments = torch.max(cluster_softmax,dim=1)
                input_predictions_list.append(cluster_assignments.cpu().numpy()) 

    input_predictions = np.concatenate(input_predictions_list,axis=0)
    for label, domain in zip(input_predictions, input_domains):
        per_domain_class_counts[label, domain] += 1
    per_domain_class_counts /= np.sum(per_domain_class_counts, axis=0, keepdims=True)
    
    prediction_outputs = np.concatenate(output_list,axis=0)
    test_labels = np.concatenate(test_labels, axis=0)
    permuted_labels = ps.get_best_permutation(prediction_outputs , test_labels) 
    
    predicted_class_prior = np.zeros_like(true_p_y_d)
    best_class_ordering = ps.best_class_ordering                
    for class_label, new_class_label in enumerate(best_class_ordering):
        predicted_class_prior[new_class_label,:] = per_domain_class_counts[class_label,:]

    reconstruction_error_L1 = np.sum(abs(true_p_y_d - predicted_class_prior))

    return label_accuracy(permuted_labels,test_labels), reconstruction_error_L1, predicted_class_prior
