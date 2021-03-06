
createDatasetFFHQ:
	 python dataset_tool.py --source=/research/datasets/ffhq/images1024x1024_test/ --dest=datasets/ffhq_test.zip

createDataset:
	 python dataset_tool.py --source=/research/boundaryCrossing/chestxray_raz/square_frontal_train --dest=datasets/xray_frontal_train.zip

#first argument contains a folder with all pngs in the same power-of-2 resolution.
createBrainDataset:
	 python dataset_tool.py --source=/razspace/brain_datasets/pngs_train --dest=datasets/brains_train_mono.zip
	
createMicroscopyDataset:
	 python dataset_tool.py create_from_images datasets/microscopy  ~/razspace/microscopy/pngs


trainBrains:
	python train.py --outdir=results --data=datasets/brains_train_mono.zip --gpus=8 --mirror=1 --cfg=paper256 


######### Image generation ####################

genFFHQ:
	python generate.py --outdir=out --trunc=1 --seeds=0,1,2 \
    --network=https://dl.dropboxusercontent.com/s/jlgybz6nfhmpv54/ffhq.pkl

######### Image reconstruction #################

# recontype =  "super-resolution", "inpaint", "k-space-cs", "all"

ffhqPkl=https://dl.dropboxusercontent.com/s/jlgybz6nfhmpv54/ffhq.pkl
xrayPkl=https://dl.dropboxusercontent.com/s/gphxjioth6dn5kb/xray.pkl
brainsPkl=https://dl.dropboxusercontent.com/s/p9vgdcn5q3wpcuo/brains.pkl
warning=-W ignore

downloadNets:
	wget https://dl.dropboxusercontent.com/s/jlgybz6nfhmpv54/ffhq.pkl
	wget https://dl.dropboxusercontent.com/s/gphxjioth6dn5kb/xray.pkl
	wget https://dl.dropboxusercontent.com/s/p9vgdcn5q3wpcuo/brains.pkl

reconFFHQ:
	python -W ignore bayesmap_recon.py --inputdir=datasets/ffhq --outdir=recFFHQ --network=ffhq.pkl --recontype=super-resolution --superres-factor 64

reconFFHQinp:
	python -W ignore bayesmap_recon.py  --inputdir=datasets/ffhq --outdir=recFFHQinp --network=ffhq.pkl --recontype=inpaint --masks=masks/1024x1024

reconXray:
	python -W ignore bayesmap_recon.py  --inputdir=datasets/xray --outdir=recXRAY --network=xray.pkl --recontype=super-resolution --superres-factor 32

reconXrayInp:
	python -W ignore bayesmap_recon.py  --inputdir=datasets/xray --outdir=recXRAYinp --network=xray.pkl --recontype=inpaint --masks=masks/1024x1024

reconBrain:
	python -W ignore bayesmap_recon.py  --inputdir=datasets/brains --outdir=recBrains --network=brains.pkl --recontype=super-resolution --superres-factor 8

reconBrainsInp:
	python -W ignore bayesmap_recon.py  --inputdir=datasets/brains --outdir=recBrainsInp --network=brains.pkl --recontype=inpaint --masks=masks/256x256

sampleFFHQ:
	python -W ignore vi_recon.py --inputdir=datasets/ffhq --outdir=samFFHQ --network=ffhq.pkl --recontype=super-resolution --superres-factor=64

sampleFFHQinp:
	python -W ignore vi_recon.py --inputdir=datasets/ffhq --outdir=samFFHQinp --network=ffhq.pkl --recontype=inpaint --masks=masks/1024x1024

sampleXray:
	python -W ignore vi_recon.py --inputdir=datasets/xray --outdir=samXRAY --network=xray.pkl --recontype=super-resolution --superres-factor=32

sampleXrayInp:
	python -W ignore vi_recon.py --inputdir=datasets/xray --outdir=samXRAYinp --network=xray.pkl --recontype=inpaint --masks=masks/1024x1024

sampleBrains:
	python -W ignore vi_recon.py --inputdir=datasets/brains --outdir=samBrains --network=brains.pkl --recontype=super-resolution --superres-factor=8

sampleBrainsInp:
	python -W ignore vi_recon.py --inputdir=datasets/brains --outdir=samBrainsInp --network=brains.pkl --recontype=inpaint --masks=masks/256x256


