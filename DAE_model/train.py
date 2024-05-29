from network import *

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_type', type=str, default="image") #image,seg
    parser.add_argument('--train_images_file', type=str, default="")
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-3)

    parser.add_argument('--input_size', type=int, default=(240,240,64))
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--results_dir', type=str, default="")
    parser.add_argument('--save_model_path', type=str,default="dae_pth")
    parser.add_argument('--ref_img_path', type=str,default="")
    args = parser.parse_args()

    ref_img = sitk.ReadImage(args.ref_img_path)
    encoder = Encoder().to(args.device)
    decoder = Decoder().to(args.device)

    train_files = glob(os.path.join(args.train_images_file, '*.nii.gz'))

    DS = Dataset(files_fixed = train_files)
    print("Number of training images: ", len(DS))
    DL  = Data.DataLoader(DS, batch_size=1, shuffle=True, num_workers=0, drop_last=True)
    loss_fn = torch.nn.MSELoss()

    params_to_optimize = [
        {'params': encoder.parameters()},
        {'params': decoder.parameters()}
    ]
    optimizer = torch.optim.Adam(params_to_optimize, lr=args.lr, weight_decay=1e-05)

    for i in range(500):  #100个epoch
        j = 0
        for noise_img,ref_image in DL:
            img = noise_img.to(device).float()
            ref =  ref_image.to(device).float()

            x = encoder(img) #1,512
            x = decoder(x)
            loss = loss_fn(x, ref)
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print('\t %d: train loss : %f' % (j,loss.data))
            j =  j + 1

        if i % 100 == 0:
                # Save model checkpoint
                save_file_name = os.path.join(args.save_model_path, f'encoder_{args.image_type}_%d.pth' % i)
                torch.save(encoder.state_dict(), save_file_name)
                save_file_name_decoder = os.path.join(args.save_model_path, f'decoder_{args.image_type}_%d.pth' % i)
                torch.save(decoder.state_dict(), save_file_name_decoder)

                # Save images
                rec_name = str(i) + "_rec.nii.gz"
                ori_name = str(i) + "_ori.nii.gz"
                save_image(x, ref_img, rec_name)
                save_image(img, ref_img, ori_name)

if __name__=="__main__":
    train()