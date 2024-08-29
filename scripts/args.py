parser = argparse.ArgumentParser()
# Feature extraction method see readme.
    # parser.add_argument('--node_entity_matching_path', type=str, default=r"./data/aminer_data/entityid_text_label.txt")
    parser.add_argument('--multimedia_feature_path', type=str,default=r'./data/fakesv_data/node_features.npy')
    parser.add_argument('--text_feature_path', type=str,default=r'./data/fakesv_data/text_features.npy')
    parser.add_argument('--ocr_feature_path', type=str,default=r'./data/fakesv_data/ocr_features.npy')
    parser.add_argument('--image_feature_path', type=str,default=r'./data/fakesv_data/image_features.npy')
    parser.add_argument('--audio_feature_path', type=str,default=r'./data/fakesv_data/audio_features.npy')
    parser.add_argument('--video_feature_path', type=str,default=r'./data/fakesv_data/video_features.npy')
    parser.add_argument('--comment_feature_path', type=str,default=r'./data/fakesv_data/comment_features.npy')
    # parser.add_argument('--text_feature_path', type=str,default=r'./data/fakesv_data/text_features.npy')
    # parser.add_argument('--feature_path', type=str,default=r'./data/aminer_data/keyword_feature.txt')
    parser.add_argument('--relation_path', type=str,default=r'./data/fakesv_data/relations.txt')
    parser.add_argument('--pos_path', type=str, default=r'./data/fakesv_data/pos_9_edges.txt')
    parser.add_argument('--id_label_path', type=str, default=r'./data/fakesv_data/id_labels.csv')

    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Disables CUDA training.')
    parser.add_argument('--fastmode', action='store_true', default=False,
                        help='Validate during training pass.')
    parser.add_argument('--seed', type=int, default=2024, help='Random seed.')
    parser.add_argument('--pretrain_epochs', type=int, default=1,
                        help='Number of epochs to train.')
    parser.add_argument('--ft_epochs', type=int, default=500,
                        help='Number of epochs to train during fine-tuning.')
    parser.add_argument('--lr', type=float, default=0.0005,
                        help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=0.001,
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--hidden', type=int, default=128,
                        help='Number of hidden units.')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate (1 - keep probability).')
    parser.add_argument('--node_dropout', type=float, default=0.1,
                        help='Dropout rate (1 - keep probability).')
    parser.add_argument('--prune', default=True, help='network pruning for model pre-training')
    parser.add_argument('--prune_ratio', type=float, default=0.2, help='network pruning for model fine-tuning')

    parser.add_argument('--number_samples', type=int, default=1000,
                        help='number of samples in co-modality pre-training')
    parser.add_argument('--device', type=str, default=torch.device("cuda:2" if torch.cuda.is_available() else "cpu"),
                        help='use GPU')
    parser.add_argument('--finetune_device', type=str, default=torch.device("cuda:2" if torch.cuda.is_available() else "cpu"),
                        help='use GPU')
    parser.add_argument('--image_encoder_model', type=str, default='swin_small_patch4_window7_224',
                        help='resnet50, please refer to for more models')
    parser.add_argument('--text_encoder_model', type=str, default='bert-base-chinese',
                        help='[bert-base-uncased,distilbert-base-uncased]')

    parser.add_argument('--node_encoder_model', type=str, default='gcn', help='[gcn,gat,sage,gin]')

    parser.add_argument('--nheads', type=int, default=8, help='gat')
    parser.add_argument('--alpha', type=float, default=0.2, help='gat')
    parser.add_argument('--image_size', type=int, default=224, help='the size of image')
    parser.add_argument('--imbalance_setting', type=str, default='focal', help='[reweight,upsample,focal]')
    parser.add_argument('--imbalance_up_scale', type=float, default=10.0, help='the scale for upsampling')

    parser.add_argument('--imbalance_ratio', type=float, default= 0.999,
                        help='[0.01, 0.1, 0, 1] if 0 then original imbalance ratio if 1 then balanced data')
    parser.add_argument('--node_feature_dim', type=int, default=768, help='the dimension of aminer feature')
    parser.add_argument('--node_feature_project_dim', type=int, default=200, help='the dimension of projected feature')
    parser.add_argument('--node_embedding_dim', type=int, default=200, help='the dimension of node embedding')
    parser.add_argument('--hidden_dim', type=int, default=200)
    parser.add_argument('--out_dim', type=int, default=200)
    parser.add_argument('--nclass', type=int, default=2, help='the number of class')

    parser.add_argument('--num_projection_layers', type=int, default=1,
                        help='the number of project layer for text/image and node')
    parser.add_argument('--feat_projection_dim', type=int, default=200,
                        help='the dimension of projected feature for nodes')
    parser.add_argument('--projection_dim', type=int, default=256,
                        help='the dimension of projected embedding of text/image and node')
    parser.add_argument('--pos', default=True)
    parser.add_argument('--debug', default=False)
    parser.add_argument('--batch_size', type=int, default=100,help='the size for each batch for co-modality pretraining')
    parser.add_argument('--num_workers', type=int, default=1, help='the number of workers')
    parser.add_argument('--patience', type=int, default=5, help='the number of epochs that the metric is not improved')
    parser.add_argument('--factor', type=int, default=0.5, help='the factor to change the learning rate')
    parser.add_argument('--temperature', type=int, default=0.1, help='the factor to change the learning rate')
    parser.add_argument('--pretrained', default=False, help="for text/image encoder")
    parser.add_argument('--trainable', default=False, help="for text/image encoder")
    parser.add_argument('--image_embedding_dim', type=int, default=768, help='the dimension of image embedding')
    parser.add_argument('--text_embedding_dim', type=int, default=768, help='the dimension of text embedding')
    parser.add_argument('--finetune_embedding_dim', type=int, default=768, help='the dimension of finetuen feature')
    parser.add_argument('--focal_alpha', type=int, default=1.0, help='focal loss')
    parser.add_argument('--focal_gamma', type=int, default=0.0, help='focal loss')
    parser.add_argument('--finetune', default=True, help='finetune the model')
