import scipy.io as sio
from os.path import join
from utils.dataset import get_dataset
from models import get_optimizer
from models.net import *
from utils.utils import *
import itertools
from scipy.io import loadmat
from noise_network import *
import time

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

criterion = nn.MSELoss()
save_path = "./result"

def get_model(config):

    configC = config.model.modelC
    configCH = config.model.modelCH
    configP = config.model.modelP

    N2N = NoiseNetwork().to(config.device)
    CNN_varphi = CovDecoder(configC).to(config.device)
    CNN_varphi_H = CovDecoder(configCH).to(config.device)
    CNN_psi = CovDecoder(configP).to(config.device)
    return N2N, CNN_varphi, CNN_varphi_H, CNN_psi

class Runner():
    def __init__(self, args, config):
        self.args = args
        self.config = config

    def test(self):
        for round in range(self.config.testing.n_round):
            dataloader = get_dataset(self.config, 'test')
            n_tracker = self.config.testing.n_tracker
            N2N, CNN_varphi, CNN_varphi_H, CNN_psi = get_model(self.config)
            init_weights(N2N, init_type='normal', init_gain=1e-2)
            init_weights(CNN_varphi, init_type='normal', init_gain=1e-5)
            init_weights(CNN_varphi_H, init_type='normal', init_gain=1e-5)
            init_weights(CNN_psi, init_type='normal', init_gain=1e-5)
            paramst = N2N.parameters()
            paramsv = CNN_varphi.parameters()
            paramsvh = CNN_varphi_H.parameters()
            paramsp = CNN_psi.parameters()

            params = itertools.chain(paramst, paramsv)
            params = itertools.chain(params, paramsvh)
            params = itertools.chain(params, paramsp)

            optimizer = get_optimizer(self.config, params)

            phi_size = self.config.testing.phi_size
            kernel_size = self.config.testing.kernel_size

            for index, point in enumerate(dataloader):
                _, atb, mask, filt = point
                atb = atb.type(torch.FloatTensor).to(self.config.device)
                s = atb.shape
                ns = int(s[1]/4)

                B = loadmat('./mask/mask_1DRU_320x320_R3.mat')
                mask1 = B['mask']
                # Obtain the validation region
                masktrain, maskval1 = mask_divide(mask1, 10)
                _, maskval2 = mask_divide(masktrain, 9)
                maskval = maskval1 + maskval2
                masktrain = mask_tile(masktrain, ns)
                maskval = mask_tile(maskval, ns)
                masktrain = torch.Tensor(masktrain)
                maskval = torch.Tensor(maskval)
                masktrain = masktrain.to(self.config.device)
                maskval = maskval.to(self.config.device)

                mask = mask.to(self.config.device)
                filt = filt.to(self.config.device)
                eta = atb.to(self.config.device)
                init_seeds(25)
                zeta = torch.randn([1, 2, 3, 3]).to(self.config.device)
                T = 0
                lowest_val_loss = np.inf
                val_loss_tracker = 0

                for epoch in range(self.config.testing.n_epochs):
                    t_start = time.time()
                    hat_z_H = N2N(eta)
                    hat_z_H = c2r(r2c(hat_z_H)/filt)
                    z_h = c2r(ifft2c(r2c(hat_z_H)))
                    tv = TV(Abs(z_h),'L1')
                    hat_exp_phi = CNN_psi(zeta)
                    hat_z = FCNN(hat_z_H, hat_exp_phi, phi_size)
                    hat_csm = CNN_varphi(zeta)
                    hat_csm_H = CNN_varphi_H(zeta)
                    hat_x = FCNN(hat_z,hat_csm,kernel_size)
                    hat_x_H = FCNN(hat_z_H,hat_csm_H,kernel_size)
                    hat_x = c2r(torch.cat([r2c(hat_x),r2c(hat_x_H)],1))
                    predicate = c2r(r2c(hat_x)*masktrain*filt)
                    true = c2r(r2c(atb)*filt*masktrain)
                    train_loss = criterion(predicate, true) + 0.00007*tv
                    optimizer.zero_grad()
                    train_loss.backward()
                    optimizer.step()

                    val_predicate = c2r(r2c(hat_x)*mask*maskval*filt)
                    val_true = c2r(r2c(atb)*filt*maskval)
                    val_loss = criterion(val_predicate, val_true)


                    if epoch >= 1000:
                        if val_loss_tracker < n_tracker or val_loss_tracker > n_tracker:
                            if val_loss <= lowest_val_loss:
                                lowest_val_loss = val_loss
                                val_loss_tracker = 0
                            else:
                                val_loss_tracker += 1
                            t_end = time.time()
                            T = T + t_end - t_start
                            out = r2c(hat_x) * (1 - mask) + r2c(atb)
                            out = out[:, :ns, :, :]
                            out = out.cpu().data.numpy()
                            out = np.transpose(out[0], [1, 2, 0])

                            out_nodc = r2c(hat_x)
                            out_nodc1 = out_nodc[:, :ns, :, :]
                            out_nodc_np = out_nodc1.cpu().data.numpy()
                            out_nodc_np = np.transpose(out_nodc_np[0], [1, 2, 0])

                            print('ROUND %d:  EPOCH %d:  TRAIN_LOSS: %.8f  VALIDATION_LOSS: %.8f  LOWEST: %.8f  TRACKER %d:'
                                  % (round, epoch, train_loss.item(), val_loss.item(), lowest_val_loss, val_loss_tracker))

                        else:
                            if val_loss <= lowest_val_loss:
                                lowest_val_loss = val_loss
                                val_loss_tracker = 0
                            else:
                                val_loss_tracker += 1
                            t_end = time.time()
                            T = T + t_end - t_start
                            out = r2c(hat_x) * (1 - mask) + r2c(atb)
                            out = out[:, :ns, :, :]
                            out = out.cpu().data.numpy()
                            out = np.transpose(out[0], [1, 2, 0])

                            out_nodc = r2c(hat_x)
                            out_nodc1 = out_nodc[:, :ns, :, :]
                            out_nodc_np = out_nodc1.cpu().data.numpy()
                            out_nodc_np = np.transpose(out_nodc_np[0], [1, 2, 0])

                            print('ROUND %d:  EPOCH %d:  TRAIN_LOSS: %.8f  VALIDATION_LOSS: %.8f  LOWEST: %.8f  TRACKER %d:'
                                % (round, epoch, train_loss.item(), val_loss.item(), lowest_val_loss, val_loss_tracker))
                            sio.savemat(join(save_path, 'KUSNNS_1DRU_R3-STOP100_dc-%d-%d.mat'% (round+1,epoch+1)), {'recon': out})
                            sio.savemat(join(save_path, 'KUSNNS_1DRU_R3-STOP100_nodc-%d-%d.mat'% (round+1,epoch+1)), {'recon': out_nodc_np})
                            break

                    else:
                        t_end = time.time()
                        T = T + t_end - t_start
                        out = r2c(hat_x) * (1 - mask) + r2c(atb)
                        out = out[:, :ns, :, :]
                        out = out.cpu().data.numpy()
                        out = np.transpose(out[0], [1, 2, 0])

                        out_nodc = r2c(hat_x)
                        out_nodc1 = out_nodc[:, :ns, :, :]
                        out_nodc_np = out_nodc1.cpu().data.numpy()
                        out_nodc_np = np.transpose(out_nodc_np[0], [1, 2, 0])

                        print('ROUND %d:  EPOCH %d:  TRAIN_LOSS: %.8f  VALIDATION_LOSS: %.8f  LOWEST: %.8f  TRACKER %d:'
                            % (round, epoch, train_loss.item(), val_loss.item(), lowest_val_loss, val_loss_tracker))
                sio.savemat(join(save_path, 'KUSNNS_1DRU_R3-10000_dc-%d.mat' % (round +1)),{'recon': out})
                sio.savemat(join(save_path, 'KUSNNS_1DRU_R3-10000_nodc-%d.mat' % (round +1)),{'recon': out_nodc_np})


