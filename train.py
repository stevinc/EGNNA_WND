import logging
import time

import numpy as np
import torch
from tqdm import tqdm
import pandas as pd


def train(model, train_loader, loss_fn, optimizer, device, params, metrics):
    start_time = time.time()
    # SET THE MODEL TO TRAIN MODE
    model.train()

    train_loss = 0.

    with tqdm(total=len(train_loader)) as t:
        for batch_idx, (in_bands, labels, blocks_adj_matrix, neighbours_numbers, points, dates, inf_dates, CoordAcc) in enumerate(train_loader):
            # move input data to GPU
            in_bands = in_bands.to(device)
            labels = labels.to(device)
            blocks_adj_matrix = blocks_adj_matrix.to(device)

            # set the gradient to zero
            optimizer.zero_grad()

            # FORWARD PASS
            out = model(in_bands, blocks_adj_matrix, neighbours_numbers)

            # Cross-Entropy classification loss
            if params.weighted_loss:
                loss = loss_fn(out, labels)
                # loss = loss * accuracy_point
                loss = loss.mean()
            else:
                loss = loss_fn(out, labels)

            val_max, preds = torch.max(out, 1)
            # probability of the positive class
            val_softmax = torch.nn.Softmax(dim=1)(out).detach()

            # BACKWARD PASS
            loss.backward()

            train_loss += loss.item()

            # write loss
            t.set_postfix(loss='{:05.3f}'.format(loss.item()))
            t.update()

            # update the params  of the model
            optimizer.step()

            # metrics... ? this code sucks..
            if batch_idx == 0:
                array_pred = preds.cpu().numpy()
                array_labels = labels.cpu().numpy()
                array_softmax = val_softmax.cpu().numpy()

                # lists_to_save = [points, inf_dates, dates, list(array_labels), list(array_pred),
                #                  list(val_softmax[:, 0].cpu().numpy()), list(val_softmax[:, 1].cpu().numpy()), CoordAcc]
                # df_to_save = pd.DataFrame(lists_to_save).transpose()
                # df_to_save.columns = ['company_cod', 'InfectionDate', 'date_image', 'labels', 'pred', 'softmax_cls0',
                #                       'softmax_cls1', 'CoordAcc']
            else:
                array_pred = np.concatenate((array_pred, preds.cpu().numpy()), axis=0)
                array_labels = np.concatenate((array_labels, labels.cpu().numpy()), axis=0)
                array_softmax = np.concatenate((array_softmax, val_softmax.cpu().numpy()), axis=0)

                # lists_to_save = [points, inf_dates, dates, list(labels.cpu().numpy()), list(preds.cpu().numpy()),
                #                  list(val_softmax[:, 0].cpu().numpy()), list(val_softmax[:, 1].cpu().numpy()), CoordAcc]
                # df_to_save_append = pd.DataFrame(lists_to_save).transpose()
                # df_to_save_append.columns = ['company_cod', 'InfectionDate', 'dates', 'labels', 'pred', 'softmax_cls0',
                #                              'softmax_cls1', 'CoordAcc']
                # df_to_save = df_to_save.append(df_to_save_append)

        # overall metrics for the current epoch, log on file
        metrics_calc = {metric: metrics[metric](array_labels, array_pred) for metric in metrics}
        metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_calc.items())
        logging.info("- Train metrics : " + metrics_string)

        # if params.name_excel:
        #     excel_version= "last"
        #     df_to_save.to_excel(params.name_excel + excel_version)

        time_elapsed = time.time() - start_time
        logging.info('Epoch complete in {:.0f}m {:.0f}s. Avg training loss: {:05.3f}'.format(
            time_elapsed // 60, time_elapsed % 60, train_loss / len(train_loader)))
