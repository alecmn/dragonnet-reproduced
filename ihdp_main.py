# from experiment.models import *
from experiment.models2 import *
import os
import glob
import argparse
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from experiment.idhp_data import *


def _split_output(yt_hat, t, y, y_scaler, x, index):
    yt_hat = yt_hat.detach().numpy()
    q_t0 = y_scaler.inverse_transform(yt_hat[:, 0].reshape(-1, 1).copy())
    q_t1 = y_scaler.inverse_transform(yt_hat[:, 1].reshape(-1, 1).copy())
    g = yt_hat[:, 2].copy()

    if yt_hat.shape[1] == 4:
        eps = yt_hat[:, 3][0]
    else:
        eps = np.zeros_like(yt_hat[:, 2])

    y = y_scaler.inverse_transform(y.copy())
    var = "average propensity for treated: {} and untreated: {}".format(g[t.squeeze() == 1.].mean(),
                                                                        g[t.squeeze() == 0.].mean())
    print(var)

    return {'q_t0': q_t0, 'q_t1': q_t1, 'g': g, 't': t, 'y': y, 'x': x, 'index': index, 'eps': eps}


def train(train_loader, net, optimizer, criterion):
    """
    Trains network for one epoch in batches.

    Args:
        train_loader: Data loader for training set.
        net: Neural network model.
        optimizer: Optimizer (e.g. SGD).
        criterion: Loss function (e.g. cross-entropy loss).
    """

    avg_loss = 0
    correct = 0
    total = 0

    # iterate through batches
    for i, data in enumerate(train_loader):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # torch.autograd.set_detect_anomaly(True)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        # print(outputs)
        # loss = criterion(labels, outputs)
        # print(loss)
        reg = 0
        reg_lambda = 1
        for param in net.parameters():
            reg += 0.5 * (param ** 2).sum()  # you can replace it with abs().sum() to get L1 regularization
        loss = criterion(outputs, labels) + reg_lambda * reg  # make the regularization part of the loss
        # print(loss)
        loss.backward()
        # for name, param in net.named_parameters():
        #     print(name, torch.isnan(param.grad).any())
        # print(param.grad)
        # nn.utils.clip_grad_norm_(net.parameters(), clip_value=5.0)
        optimizer.step()

        # keep track of loss and accuracy
        avg_loss += loss
        # _, predicted = torch.max(outputs.data, 1)
        # total += labels.size(0)
        # correct += (predicted == labels).sum().item()

    return avg_loss / len(train_loader)  # , 100 * correct / total


def test(test_loader, net, criterion):
    """
    Evaluates network in batches.

    Args:
        test_loader: Data loader for test set.
        net: Neural network model.
        criterion: Loss function (e.g. cross-entropy loss).
    """

    avg_loss = 0
    correct = 0
    total = 0

    # Use torch.no_grad to skip gradient calculation, not needed for evaluation
    with torch.no_grad():
        # iterate through batches
        for data in test_loader:
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # forward pass
            outputs = net(inputs)
            loss1 = criterion(outputs[0], labels)
            loss2 = criterion(outputs[1], labels)
            loss3 = criterion(outputs[2], labels)
            loss4 = criterion(outputs[3], labels)
            loss = loss1 + loss2 + loss3 + loss4

            # keep track of loss and accuracy
            avg_loss += loss
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return avg_loss / len(test_loader), 100 * correct / total


def normalize(x):
    x_normed = x / x.max(0, keepdim=True)[0]
    return x_normed


def train_and_predict_dragons(t, y_unscaled, x, targeted_regularization=True, output_dir='',
                              knob_loss=dragonnet_loss_binarycross, ratio=1., dragon='', val_split=0.2, batch_size=64):
    verbose = 0
    y_scaler = StandardScaler()
    y = y_scaler.fit_transform(y_unscaled)
    train_outputs = []
    test_outputs = []

    # if dragon == 'tarnet':
    #     dragonnet = make_tarnet(x.shape[1], 0.01)

    # elif dragon == 'dragonnet':
    print("I am here making dragonnet")
    dragonnet = DragonNet(x.shape[1])

    # metrics = [regression_loss, binary_classification_loss, treatment_accuracy, track_epsilon]
    #
    # if targeted_regularization:
    #     loss = make_tarreg_loss(ratio=ratio, dragonnet_loss=knob_loss)
    # else:
    #     loss = knob_loss

    loss = knob_loss
    # for reporducing the IHDP experimemt

    i = 0
    torch.manual_seed(i)
    np.random.seed(i)
    # train_index, test_index = train_test_split(np.arange(x.shape[0]), test_size=0., random_state=1)
    train_index = np.arange(x.shape[0])
    test_index = train_index

    x_train, x_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]
    t_train, t_test = t[train_index], t[test_index]

    yt_train = np.concatenate([y_train, t_train], 1)

    tensors_train = torch.from_numpy(x_train).float(), torch.from_numpy(yt_train).float()
    tensors_test = torch.from_numpy(x_test).float(), torch.from_numpy(y_test).float()

    train_loader = DataLoader(TensorDataset(*tensors_train), batch_size=100)
    test_loader = DataLoader(TensorDataset(*tensors_test), batch_size=100)

    import time;
    start_time = time.time()

    epochs = 100

    optimizer_Adam = optim.Adam([{'params': dragonnet.representation_block.parameters()},
                                 {'params': dragonnet.t_predictions.parameters()},
                                 {'params': dragonnet.t0_head.parameters(), 'weight_decay': 0.01},
                                 {'params': dragonnet.t1_head.parameters(), 'weight_decay': 0.01}], lr=1e-6)
    optimizer_SGD = optim.SGD([{'params': dragonnet.representation_block.parameters()},
                               {'params': dragonnet.t_predictions.parameters()},
                               {'params': dragonnet.t0_head.parameters(), 'weight_decay': 0.01},
                               {'params': dragonnet.t1_head.parameters(), 'weight_decay': 0.01}], lr=1e-6, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer_Adam, 'min')

    for epoch in tqdm(range(epochs)):
        # Train on data
        train_loss = train(train_loader, dragonnet, optimizer_Adam, loss)
        print(f"Epoch: {epoch + 1}, loss: {train_loss}")

        scheduler.step(train_loss)

        # # Test on data
        # test_loss, test_acc = test(test_loader, dragonnet, loss)
    for epoch in tqdm(range(epochs)):
        # Train on data
        train_loss = train(train_loader, dragonnet, optimizer_SGD, loss)
        print(f"Epoch: {epoch + 1}, loss: {train_loss}")

    elapsed_time = time.time() - start_time
    print("***************************** elapsed_time is: ", elapsed_time)

    # yt_hat_test = dragonnet.predict(x_test)
    # yt_hat_train = dragonnet.predict(x_train)

    yt_hat_test = dragonnet(torch.from_numpy(x_test).float())
    yt_hat_train = dragonnet(torch.from_numpy(x_train).float())

    test_outputs += [_split_output(yt_hat_test, t_test, y_test, y_scaler, x_test, test_index)]
    train_outputs += [_split_output(yt_hat_train, t_train, y_train, y_scaler, x_train, train_index)]

    return test_outputs, train_outputs
