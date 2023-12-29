


def data_process(WSI_name_list, sur_time_list, censor_list):
    list_data = []
    for i in range(len(WSI_name_list)):
        list_data.append((censor_list[i], sur_time_list[i], WSI_name_list[i]))
    list_data = sorted(list_data)
    for idx, i in enumerate(list_data):
        censor_list[idx], sur_time_list[idx], WSI_name_list[idx] = i
    prt = []
    index_all = []
    for i in range(len(WSI_name_list)):
        prt.append(i)
        if i % 5 == 4 or i == len(WSI_name_list) - 1:
            random.shuffle(prt)
            index_all.extend(prt)
            prt = []
    return WSI_name_list, sur_time_list, censor_list, index_all


def get_one_fold_index(fold_num, number_kfold, index_all):
    train_index_split = []
    val_index_split = []
    test_index = []
    for i in range(len(WSI_name_list)):
        if i % number_kfold == fold_num:
            test_index.append(index_all[i])
        elif i % number_kfold == ((fold_num + 1) % number_kfold):
            val_index_split.append(index_all[i])
        else:
            train_index_split.append(index_all[i])
    return train_index_split, val_index_split, test_index


def once_run(args, WSI_name_list, sur_time_list, censor_list, seed, run_num, data_map):
    all_fold_test_loss = []
    all_fold_test_assessment = []
    all_fold_test_assessment_cmp = []
    all_fold_best_val_assessment = []
    WSI_name_list, sur_time_list, censor_list, index_all = data_process(WSI_name_list, sur_time_list, censor_list)
    for fold_num in range(5):
        train_index_split, val_index_split, test_index = get_one_fold_index(fold_num, args.number_kfold, index_all)
        best_assessment_val_one_fold = [0, -1]
        best_model_path_one_fold = ''
        args_cmp = copy.deepcopy(args)
        args_cmp.number_scale = 1
        model = PP_MGCN.MIL(args, args_cmp)
        model = model.cuda()
        optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        print('--------------------------------------------------------------------------------------------')
        print(model)
        print('--------------------------------------------------------------------------------------------')
        print(train_index_split, '\n', val_index_split, '\n', test_index)
        print('--------------------------------------------------------------------------------------------')
        print("dataset:", args.dataset, "  model:", args.model, "  task:", args.task, "  gpu:", gpu_ids, "  seed:",
              args.divide_seed,
              "  lr:", args.lr, "  batch_size:", args.batch_size, "  patch_size:", args.patch_size, "   number_scale:",
              args.number_scale,
              "  using_Swin:", args.using_Swin, "   gcn_layer", args.gcn_layer)
        epoch_no_update = 0
        for epoch_num in range(args.epochs):
            if epoch_no_update == args.epochs_patience:
                break
            train_loss, train_assessment, train_assessment_cmp = train(args, args_cmp, model, train_index_split,
                                                                       WSI_name_list,
                                                                       sur_time_list, censor_list, optimizer,
                                                                       epoch_num + 1, fold_num + 1, run_num, data_map)
            val_loss, val_assessment, val_assessment_cmp = val_and_test(args, args_cmp, model, val_index_split,
                                                                        WSI_name_list, sur_time_list,
                                                                        censor_list,
                                                                        epoch_num + 1, fold_num + 1, run_num, data_map)
            print("   epoch：{:2d}  train_loss：{:.4f}   val_loss：{:.4f} ".format(epoch_num + 1, train_loss, val_loss),
                  "\n  train assessment:", train_assessment,
                  "\n  train assessment_cmp:", train_assessment_cmp,
                  "\n  val_assessment:", val_assessment,
                  "\n  val_assessment_cmp:", val_assessment_cmp
                  )

            if epoch_num < args.epochs_warm:
                print("   warm epoch!")
            elif (val_assessment) >= best_assessment_val_one_fold[0]:
                best_assessment_val_one_fold = [val_assessment, epoch_num]
                if best_model_path_one_fold != '' and os.path.exists(best_model_path_one_fold):
                    os.remove(best_model_path_one_fold)
                    print("   delete last model")
                best_model_path_one_fold = os.path.join(args.model_save_path,
                                                        args.dataset + '_' + args.task + '_' + args.model +
                                                        '_' + str(fold_num + 1) + '_' + args.time_stamp + '.pth')
                os.makedirs(args.model_save_path, exist_ok=True)
                torch.save(model.state_dict(), best_model_path_one_fold)
                print("   model saved in", best_model_path_one_fold)
                epoch_no_update = 0
            else:
                epoch_no_update = epoch_no_update + 1

        model_test = PP_MGCN.MIL(args, args_cmp)
        model_test = model_test.cuda()
        model_test.load_state_dict(torch.load(best_model_path_one_fold))
        model_test.eval()
        test_loss, test_assessment, test_assessment_cmp = val_and_test(args, args_cmp, model_test, test_index,
                                                                       WSI_name_list, sur_time_list,
                                                                       censor_list, -1, fold_num + 1, run_num, data_map)
        all_fold_test_loss.append(test_loss)
        all_fold_test_assessment.append(test_assessment)
        all_fold_test_assessment_cmp.append(test_assessment_cmp)
        all_fold_best_val_assessment.append(best_assessment_val_one_fold)
        print('  test_loss:', test_loss)
        print('  test assessment:', test_assessment)
        print('  best_val_assessment:', best_assessment_val_one_fold)
        print('  test assessment_cmp:', test_assessment_cmp)

    print(' run:', run_num, ' All fold test loss:', all_fold_test_loss, ',mean:', np.mean(all_fold_test_loss))
    print(' run:', run_num, ' All fold best val assessment:', all_fold_best_val_assessment)
    print(' run:', run_num, ' All fold test assessment:', all_fold_test_assessment)
    print(' run:', run_num, ' All fold test assessment_cmp:', all_fold_test_assessment_cmp)
    print(' run:', run_num, "mean c-index:", np.mean(all_fold_test_assessment))
    print(' run:', run_num, "mean c-index cmp:", np.mean(all_fold_test_assessment_cmp))
    return np.mean(all_fold_test_assessment)


def get_Y(prediction):
    S = 1.0
    Y = 0.0
    for ii in prediction[0]:
        S = S * ii
        Y = Y + S
    Y = Y / len(prediction[0])
    return Y


def train(args, args_cmp, model, index_split, WSI_name_list, sur_time_list, censor_list, optimizer, epoch_num,
          fold_num, run_num, data_map):
    model.train()
    random.shuffle(index_split)
    total_loss = 0
    Y_list = []
    Y_list_cmp = []
    ass_sur_time_list = []
    ass_sur_time_list_cmp = []
    ass_censor_list = []
    ass_censor_list_cmp = []
    for (i, idx) in enumerate(index_split):
        feats_list, sur_time, censor, edge_index, edge_index_diff, feats_size_list, feats_info = \
            utils.sur_bag_build.get_bag(args, WSI_name_list[idx], sur_time_list[idx], censor_list[idx], data_map)
        feats_list_cmp, sur_time_cmp, censor_cmp, edge_index_cmp, edge_index_diff_cmp, feats_size_list_cmp, feats_info_cmp = \
            utils.sur_bag_build.get_bag(args_cmp, WSI_name_list[idx], sur_time_list[idx], censor_list[idx], data_map)

        prediction, at_, prediction_cmp, at_cpm = model(feats_list, edge_index, edge_index_diff, feats_size_list,
                                                        feats_list_cmp, edge_index_cmp, edge_index_diff_cmp,
                                                        feats_size_list_cmp, args.mask_prob)

        loss = utils.sur_loss.sur_loss_cc(prediction, prediction_cmp, sur_time, censor)
        loss_cpu = loss.item()
        total_loss += (loss_cpu)
        loss.backward()

        Y = get_Y(prediction)
        Y_cmp = get_Y(prediction_cmp)
        Y_list.append(Y)
        Y_list_cmp.append(Y_cmp)
        ass_sur_time_list.append(sur_time)
        ass_censor_list.append(censor)
        ass_sur_time_list_cmp.append(sur_time_cmp)
        ass_censor_list_cmp.append(censor_cmp)
        if i % (len(index_split) // 20) == 0:  # out 20 case
            print(
                "    run/fold/epoch：{:d}/{:d}/{:d}  {:d}/{:d}   now_loss：{:.4f}   censor：{:.0f}   sur_time：{:.0f}   Y：{:.4f}    predict：{:.4f}  ".format(
                    run_num,
                    fold_num, epoch_num, i + 1, len(index_split), loss_cpu, censor_list[idx], sur_time_list[idx],
                    Y.item(), prediction[0][int(sur_time_list[idx])].item()), feats_list.shape
            )
            for ii in prediction[0]:
                print(round(ii.item(), 2), end=' ')
            print()
        if (i + 1) % args.batch_size == 0 or (i + 1) == len(index_split):
            optimizer.step()
            optimizer.zero_grad()

    c_index = utils.sur_estimate.c_index_cal(Y_list, ass_sur_time_list, ass_censor_list)
    c_index_cmp = utils.sur_estimate.c_index_cal(Y_list_cmp, ass_sur_time_list_cmp, ass_censor_list_cmp)
    return total_loss / len(index_split), c_index, c_index_cmp


def val_and_test(args, args_cmp, model, index_split, WSI_name_list, sur_time_list, censor_list, epoch_num, fold_num,
                 run_num,
                 data_map):
    model.eval()
    random.shuffle(index_split)
    with torch.no_grad():
        total_loss = 0
        Y_list = []
        Y_list_cmp = []
        ass_sur_time_list = []
        ass_sur_time_list_cmp = []
        ass_censor_list = []
        ass_censor_list_cmp = []
        for (i, idx) in enumerate(index_split):
            feats_list, sur_time, censor, edge_index, edge_index_diff, feats_size_list, feats_info = \
                utils.sur_bag_build.get_bag(args, WSI_name_list[idx], sur_time_list[idx], censor_list[idx], data_map)
            feats_list_cmp, sur_time_cmp, censor_cmp, edge_index_cmp, edge_index_diff_cmp, feats_size_list_cmp, feats_info_cmp = \
                utils.sur_bag_build.get_bag(args_cmp, WSI_name_list[idx], sur_time_list[idx], censor_list[idx],
                                            data_map)

            prediction, at_, prediction_cmp, at_cpm = model(feats_list, edge_index, edge_index_diff, feats_size_list,
                                                            feats_list_cmp, edge_index_cmp, edge_index_diff_cmp,
                                                            feats_size_list_cmp, args.mask_prob)

            loss = utils.sur_loss.sur_loss_cc(prediction, prediction_cmp, sur_time, censor)
            loss_cpu = loss.item()
            total_loss += (loss_cpu)

            Y = get_Y(prediction)
            Y_cmp = get_Y(prediction_cmp)
            Y_list.append(Y)
            Y_list_cmp.append(Y_cmp)
            ass_sur_time_list.append(sur_time)
            ass_censor_list.append(censor)
            ass_sur_time_list_cmp.append(sur_time_cmp)
            ass_censor_list_cmp.append(censor_cmp)
            if i % (len(index_split) // 20) == 0:  # out 20 case
                print(
                    "    run/fold/epoch：{:d}/{:d}/{:d}  {:d}/{:d}   now_loss：{:.4f}   censor：{:.0f}   sur_time：{:.0f}   Y：{:.4f}    predict：{:.4f}  ".format(
                        run_num,
                        fold_num, epoch_num, i + 1, len(index_split), loss_cpu, censor_list[idx], sur_time_list[idx],
                        Y.item(), prediction[0][int(sur_time_list[idx])].item()), feats_list.shape
                )
                for ii in prediction[0]:
                    print(round(ii.item(), 2), end=' ')
                print()

        c_index = utils.sur_estimate.c_index_cal(Y_list, ass_sur_time_list, ass_censor_list)
        c_index_cmp = utils.sur_estimate.c_index_cal(Y_list_cmp, ass_sur_time_list_cmp, ass_censor_list_cmp)
        return total_loss / len(index_split), c_index, c_index_cmp


if __name__ == '__main__':
    import os
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--patch_size", type=int, default=512, help="patch_size to use")
    parser.add_argument('--gpu_index', type=int, default=5, help='GPU ID(s)')
    parser.add_argument("--dataset", type=str, default="TCGA_LUAD",
                        help="Database to use[TCGA_LUAD,TCGA_LUSC,TCGA_UCEC,TCGA_BRCA,TCGA_GBMLGG,TCGA_BLCA]")
    parser.add_argument("--model", type=str, default="sur_PP_MGCN")
    parser.add_argument("--in_classes", type=int, default=1024, help="Feature size")
    parser.add_argument("--out_classes", type=int, default=30, help="Survival vector")
    # ------SWAP_GCN
    parser.add_argument("--magnification_scale", type=int, default=3, help="")
    parser.add_argument("--number_scale", type=int, default=3, help="[1,4]")
    parser.add_argument("--using_Swin", type=int, default=1, help="[0,1]")
    parser.add_argument("--gcn_layer", type=int, default=1, help="Number of graph convs in each scale")
    parser.add_argument("--mask_prob", type=float, default=0, help="")
    # -----SWAP_GCN
    parser.add_argument("--model_save_path", type=str, default="saved_model", help="path for save model")
    parser.add_argument("--task", type=str, default="survival", help="Task of classification[survival]")
    parser.add_argument("--divide_seed", type=int, default=2023, help="")
    # ------------------
    parser.add_argument("--batch_size", type=int, default=32, help="")
    parser.add_argument("--epochs", type=int, default=300, help="")
    parser.add_argument("--epochs_patience", type=int, default=64, help="")
    parser.add_argument("--epochs_warm", type=int, default=32, help="")
    parser.add_argument("--drop_out_ratio", type=float, default=0.2, help="")
    parser.add_argument("--lr", type=float, default=0.000005, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.000001, help="")
    # ------------------
    parser.add_argument("--number_kfold", type=int, default=5, help="Number of KFold")
    parser.add_argument("--number_run", type=int, default=1, help="Number of runs")
    parser.add_argument("--time_stamp", type=str, default="-1")
    args, _ = parser.parse_known_args()
    gpu_ids = tuple((args.gpu_index,))
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(x) for x in gpu_ids)
    import copy
    import sys
    import torch
    import random
    import os.path
    import numpy as np
    from torch.optim import Adam
    import datetime
    import utils.sys_utils
    import utils.sur_bag_init
    import utils.sur_bag_build
    import utils.sur_loss
    import utils.sur_estimate
    import model.sur_PP_MGCN as PP_MGCN

    args.time_stamp = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S.%f')
    sys.stdout = utils.sys_utils.Logger(sys.stdout, model_name=args.model + args.dataset + "_cmp")
    sys.stderr = utils.sys_utils.Logger(sys.stderr, model_name=args.model + args.dataset + "_cmp")
    print("dataset:", args.dataset, "  model:", args.model, "  task:", args.task, "  gpu:", gpu_ids, "  seed:",
          args.divide_seed,
          "  lr:", args.lr, "  batch_size:", args.batch_size, "  patch_size:", args.patch_size, "   number_scale:",
          args.number_scale,
          "  using_Swin:", args.using_Swin, "   gcn_layer:", args.gcn_layer, "   magnification_scale",
          args.magnification_scale, "    out_classes:", args.out_classes, "mask_prob", args.mask_prob)
    print('--------------------------------------------------------------------------------------------')
    with open('model/sur_SWAP_GCN.py', 'r') as viewFile:
        data = viewFile.read()
    print(data)
    with open('sur_main_cc.py', 'r') as viewFile:
        data = viewFile.read()
    print(data)
    print('--------------------------------------------------------------------------------------------')

    assert args.number_scale in [1, 2, 3, 4, 5]
    assert args.dataset in ['TCGA_LUAD', 'TCGA_LUSC', 'TCGA_UCEC', 'TCGA_BRCA', 'TCGA_GBMLGG', 'TCGA_BLCA']
    assert args.task in ['survival']
    WSI_name_list = []
    sur_time_list = []
    censor_list = []
    if args.task == 'survival' and args.dataset == 'TCGA_LUAD':
        WSI_name_list, sur_time_list, censor_list = utils.sur_bag_init.sur_get_tcga_luad_bags(args)
    if args.task == 'survival' and args.dataset == 'TCGA_LUSC':
        WSI_name_list, sur_time_list, censor_list = utils.sur_bag_init.sur_get_tcga_lusc_bags(args)
    if args.task == 'survival' and args.dataset == 'TCGA_UCEC':
        WSI_name_list, sur_time_list, censor_list = utils.sur_bag_init.sur_get_tcga_ucec_bags(args)
    if args.task == 'survival' and args.dataset == 'TCGA_BRCA':
        WSI_name_list, sur_time_list, censor_list = utils.sur_bag_init.sur_get_tcga_brca_idc_bags(args)
    if args.task == 'survival' and args.dataset == 'TCGA_BLCA':
        WSI_name_list, sur_time_list, censor_list = utils.sur_bag_init.sur_get_tcga_blca_bags(args)
    if args.task == 'survival' and args.dataset == 'TCGA_GBMLGG':
        WSI_name_list, sur_time_list, censor_list = utils.sur_bag_init.sur_get_tcga_gbmlgg_bags(args)
    print("time:", datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S.%f'))
    all_c_index = []
    data_map = {}
    for i in range(args.number_run):
        seed = args.divide_seed + i
        utils.sys_utils.setup_seed(seed)
        once_run_c_index = once_run(args, WSI_name_list, sur_time_list, censor_list, seed, i + 1, data_map)
        all_c_index.append(once_run_c_index)

    print("all run c-index:", all_c_index, "  mean:", np.mean(all_c_index))
    print("dataset:", args.dataset, "  model:", args.model, "  task:", args.task, "  gpu:", gpu_ids, "  seed:",
          args.divide_seed,
          "  lr:", args.lr, "  batch_size:", args.batch_size, "  patch_size:", args.patch_size, "   number_scale:",
          args.number_scale,
          "  using_Swin:", args.using_Swin, "   gcn_layer", args.gcn_layer, "   magnification_scale",
          args.magnification_scale, " mask_prob:", args.mask_prob)
