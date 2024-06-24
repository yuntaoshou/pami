## for lexical feature extraction
https://huggingface.co/microsoft/deberta-large/tree/main  -> ../tools/transformers/deberta-large

## for acoustic feature extraction
https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_large.pt  -> ../tools/wav2vec

## for face extractor (OpenFace-win)
https://drive.google.com/file/d/1-O8epcTDYCrRUU_mtXgjrS3OWA4HTp0-/view?usp=share_link  -> ./OpenFace_2.2.0_win_x64

## for visual feature extraction
https://drive.google.com/file/d/1wT2h5sz22SaEL4YTBwTIB3WoL4HUvg5B/view?usp=share_link ->  ../tools/manet

## using ffmpeg for sub-video extraction
https://ffmpeg.org/download.html#build-linux ->  ../tools/ffmpeg-4.4.1-i686-static

# download IEMOCAP dataset and put it into ../emotion-data/IEMOCAP
https://sail.usc.edu/iemocap/iemocap_release.htm   ->   ../emotion-data/IEMOCAP

# whole video -> subvideo
python preprocess.py split_video_by_start_end_IEMOCAP

# subvideo -> detect face
python detect.py --model='face_detection_yunet_2021sep.onnx' --videofolder='dataset/IEMOCAP/subvideo' --save='dataset/IEMOCAP/subvideofaces' --dataset='IEMOCAP'

# extract visual features
cd feature_extraction/visual
python extract_manet_embedding.py --dataset='IEMOCAPFour' --gpu=0
python preprocess.py feature_compressed_iemocap dataset/IEMOCAP/features/manet dataset/IEMOCAP/features/manet_UTT

# extract acoustic features
python preprocess.py split_audio_from_video_16k 'dataset/IEMOCAP/subvideo' 'dataset/IEMOCAP/subaudio'
cd feature_extraction/audio
python extract_wav2vec_embedding.py --dataset='IEMOCAPFour' --feature_level='UTTERANCE' --gpu=0

# extract textual features
python preprocess.py generate_transcription_files_IEMOCAP
cd feature_extraction/text
python extract_text_embedding_LZ.py --dataset='IEMOCAPFour' --feature_level='UTTERANCE' --model_name='deberta-large' --gpu=0

###################################################################
# We also provide pre-extracted multimodal features
IEMOCAP: https://drive.google.com/file/d/1Hn82-ZD0CNqXQtImd982YHHi-3gIX2G3/view?usp=share_link  -> ./dataset/IEMOCAP/features
CMUMOSI: https://drive.google.com/file/d/1aJxArYfZsA-uLC0sOwIkjl_0ZWxiyPxj/view?usp=share_link  -> ./dataset/CMUMOSI/features
CMUMOSEI:https://drive.google.com/file/d/1L6oDbtpFW2C4MwL5TQsEflY1WHjtv7L5/view?usp=share_link  -> ./dataset/CMUMOSEI/features

cd baseline-mmin
python change_format.py change_feat_format_iemocapsix

python -u train_miss.py --mask_rate=0.2 --dataset_mode=iemocapsix_miss  --model=mmin_AE --log_dir=./logs --checkpoints_dir=./checkpoints --print_freq=10 --gpu_ids=0 --input_dim_a=512 --embd_size_a=128 --input_dim_v=1024 --embd_size_v=128 --input_dim_l=1024 --embd_size_l=128 --AE_layers=256,128 --ce_weight=1.0 --mse_weight=0.2 --cls_layers=128,128 --dropout_rate=0.5 --niter=20 --niter_decay=80 --init_type normal --batch_size=256 --lr=1e-3 --run_idx=8 --name=mmin --suffix=iemocapsix_AETemp

## differently, CPM-Net runs in another environment, see requirements-cpmnet.txt for more details.
cd baseline-cpmnet

## change feature format
python change_format.py change_feat_format_iemocapsix

## training model
python test_lianzheng.py --dataset='iemocapsix'  --missing-rate=0.2 --epochs-train=30 --epochs-test=300 --lsd-dim=128 --lamb=1

cd baseline-cca

# training with cpmnet-generated data format
please first run ''python change_format.py change_feat_format_iemocapsix'' in baseline-cpmnet

# train CCA
python cca.py   --dataset='iemocapsix' --missing-rate=0.2 --n-components=2

# train DCCA
python dcca.py  --dataset='iemocapsix' --missing-rate=0.2 --n-hidden=256 --max-epoch=100 --n-components=2

# train DCCAE
python dccae.py --dataset='iemocapsix' --missing-rate=0.2 --n-hidden=256 --max-epoch=100 --n-components=2

