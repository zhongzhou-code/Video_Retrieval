import numpy as np
import torch
from utils_log.perf_log import logger2
from torch.autograd import Variable

def test_MAP(model, train_dataloader, test_loader):

    logger2.info('          Waiting for generate the hash code from train set')
    database_hash, database_labels = predict_hash_code(model, train_dataloader)
    logger2.info('          database_hash.shape : {} , database_labels.shape : {}'.format(database_hash.shape,  database_labels.shape))

    logger2.info('          Waiting for generate the hash code from test set')
    test_hash, test_labels = predict_hash_code(model, test_loader)
    logger2.info('          test_hash.shape : {} , test_labels.shape : {}'.format(test_hash.shape, test_labels.shape))

    logger2.info('          Come into the function of mean_average_precision')
    MAP = mean_average_precision(database_hash, test_hash, database_labels, test_labels)
    logger2.info('          ')

    return MAP

# data_loader is train_loader or test_loader
def predict_hash_code(model, data_loader):
    model.eval()
    is_start = True

    for step, (frames, label) in enumerate(data_loader):
        frames = Variable(frames).cuda()
        label = Variable(label).cuda()
        hash_features = model(frames)
        if is_start:
            all_output = hash_features.data.cpu().float()
            all_label = label.float()
            is_start = False
        else:
            all_output = torch.cat((all_output, hash_features.data.cpu().float()), 0)
            all_label = torch.cat((all_label, label.float()), 0)

    return all_output.cpu().numpy(), all_label.cpu().numpy()

def mean_average_precision(database_hash, test_hash, database_labels, test_labels):  # R = 1000
    # binary the hash code
    R = 100
    T = 0

    # Binary-like codes are mapped to binary codes
    database_hash[database_hash < T] = -1
    database_hash[database_hash >= T] = 1
    test_hash[test_hash < T] = -1
    test_hash[test_hash >= T] = 1

    # total number for testing
    query_num = test_hash.shape[0]

    # Do the dot product operation to get sim, the sim[i, j]
    sim = np.dot(database_hash, test_hash.T)
    logger2.info('              sim.shape : {}'.format(sim.shape))

    # sort function, Sort the matrix sim according to the axis = 0 and return the sorted subscripts
    ids = np.argsort(-sim, axis=0)
    logger2.info('              ids.shape : {}'.format(ids.shape))

    APx = []
    Recall = []

    # Calculate column by column
    for i in range(query_num):

        # The label corresponding to the test sample, label.shape = torch.Size([])
        label = test_labels[i]

        # binary-hash of column i, idx.shape = torch.Size([train_hash_size])
        idx = ids[:, i]
        #
        list_of_match = (database_labels[idx[0:R]] == label) > 0
        logger2.info('                  list_of_match : {}'.format(list_of_match))

        list_of_match.astype(int)
        logger2.info('                  list_of_match : {}'.format(list_of_match))

        relevant_num = np.sum(list_of_match)
        logger2.info('                  relevant_num : {}'.format(relevant_num))

        Lx = np.cumsum(list_of_match)   # Return the cumulative sum of the elements along a given axis.

        Px = Lx.astype(float) / np.arange(1, R + 1, 1)      # from 1 to 100

        if relevant_num != 0:
            APx.append(np.sum(Px * list_of_match) / relevant_num)

        if relevant_num == 0:   # even no relevant image, still need add in APx for calculating the mean
            APx.append(0)

        all_relevant = np.sum(database_labels == label, axis=1) > 0
        all_num = np.sum(all_relevant)
        r = relevant_num / np.float(all_num)
        Recall.append(r)

    return np.mean(np.array(APx)), np.mean(np.array(Recall)), APx