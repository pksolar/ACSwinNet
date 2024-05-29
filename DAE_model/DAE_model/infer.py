from network import *


def test():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_images_file', type=str, default="")
    parser.add_argument('--encoder_pth', type=str, default="")
    parser.add_argument('--decoder_pth', type=str, default="")
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-3)

    parser.add_argument('--input_size', type=int, default=(240,240,64))
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--results_dir', type=str, default="")
    parser.add_argument('--save_model', action='store_true')
    parser.add_argument('--ref_img_path', type=str, default="")
    args = parser.parse_args()

    ref_img = sitk.ReadImage(args.ref_img_path)
    encoder = Encoder()
    decoder = Decoder()

    encoder.load_state_dict(torch.load(args.encoder_pth))['state_dict']
    decoder.load_state_dict(torch.load(args.decoder_pth))['state_dict']

    encoder.to(args.device)
    decoder.to(args.device)

    train_files = glob(os.path.join(args.train_images_file, '*.nii.gz'))

    DS = Dataset(files_fixed=train_files)
    print("Number of training images: ", len(DS))
    DL = Data.DataLoader(DS, batch_size=1, shuffle=True, num_workers=0, drop_last=True)

    j = 0
    with torch.no_grad():
        for noise_img, ref_image in DL:
            img = noise_img.to(args.device).float()
            x = encoder(img)  # 1,512
            x = decoder(x)
            rec_name = str(j) + "_rec.nii.gz"
            ori_name = str(j) + "_ori.nii.gz"
            save_image(x, ref_img, rec_name)
            save_image(img, ref_img, ori_name)

if __name__=="__main__":
    test()