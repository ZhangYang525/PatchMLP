from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
from utils.dtw_metric import accelerated_dtw  # 避免名字与变量冲突
from utils.frequency_loss import FrequencyDomainL1Loss

warnings.filterwarnings('ignore')


class Exp_Long_Term_Forecast(Exp_Basic):
    """
    长期预测实验：
    - 保持与原版完全一致的外部接口：_build_model/_get_data/_select_optimizer/_select_criterion/vali/train/test
    - 内部精简重复逻辑并修复若干隐患（AMP、loss 聚合、切片一致性、加载 map_location 等）
    """

    def __init__(self, args):
        super().__init__(args)
        # 在 CPU 上强开 AMP 会带来问题，这里做个保护
        self._amp_enabled = bool(getattr(self.args, "use_amp", False) and "cuda" in str(self.device))

    # ----------------------- 内部小工具 -----------------------

    def _decoder_input(self, batch_y: torch.Tensor) -> torch.Tensor:
        """构造 decoder 输入：label_len + pred_len 的零填充。"""
        dec_zeros = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
        dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_zeros], dim=1).float()
        return dec_inp.to(self.device, non_blocking=True)

    def _slice_pred_and_true(self, outputs: torch.Tensor, batch_y: torch.Tensor):
        """
        统一的切片规则：
        - features == 'MS' 时，仅取最后一维通道（目标变量）；否则取全部通道
        - 统一在时间维上取最后 pred_len
        """
        f_dim = -1 if self.args.features == 'MS' else 0
        outputs = outputs[:, -self.args.pred_len:, f_dim:]
        batch_y = batch_y[:, -self.args.pred_len:, f_dim:]
        return outputs, batch_y

    def _forward_once(self, batch_x, batch_y, batch_x_mark, batch_y_mark):
        """单步前向，自动套 AMP，上下文内不做 .detach()/.cpu()。"""
        with torch.cuda.amp.autocast(enabled=self._amp_enabled):
            dec_inp = self._decoder_input(batch_y)
            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        return outputs

    # ----------------------- 框架要求的方法 -----------------------

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()
        if getattr(self.args, "use_multi_gpu", False) and getattr(self.args, "use_gpu", False):
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        # 如需 AdamW：请在外部 args 里切参，这里保持原接口
        return optim.Adam(self.model.parameters(), lr=self.args.learning_rate)

    def _select_criterion(self):
        if getattr(self.args, "loss", "MSE") == "MSE":
            return nn.MSELoss()
        # 避免 device 不一致产生的警告
        crit = FrequencyDomainL1Loss(alpha=0.3, freq_mode='log')
        return crit.to(self.device)

    # ----------------------- 训练/验证/测试 -----------------------

    def vali(self, vali_data, vali_loader, criterion):
        self.model.eval()
        losses = []
        with torch.no_grad():
            for (batch_x, batch_y, batch_x_mark, batch_y_mark) in vali_loader:
                batch_x = batch_x.float().to(self.device, non_blocking=True)
                batch_y = batch_y.float().to(self.device, non_blocking=True)
                batch_x_mark = batch_x_mark.float().to(self.device, non_blocking=True)
                batch_y_mark = batch_y_mark.float().to(self.device, non_blocking=True)

                outputs = self._forward_once(batch_x, batch_y, batch_x_mark, batch_y_mark)
                outputs, target = self._slice_pred_and_true(outputs, batch_y)
                loss = criterion(outputs, target)
                # 统一返回标量
                losses.append(float(loss.item()))
        self.model.train()
        return float(np.mean(losses)) if losses else float('inf')

    def train(self, setting):
        # 打印模型参数规模
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"\n>>> Model total parameters: {total_params:,} ({total_params/1e6:.2f}M)")
        print(f">>> Model trainable parameters: {trainable_params:,} ({trainable_params/1e6:.2f}M)\n")

        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        # checkpoint 目录
        ckpt_dir = os.path.join(self.args.checkpoints, setting)
        os.makedirs(ckpt_dir, exist_ok=True)

        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        scaler = torch.cuda.amp.GradScaler(enabled=self._amp_enabled)

        train_steps = len(train_loader)
        time_now = time.time()
        for epoch in range(self.args.train_epochs):
            self.model.train()
            epoch_start = time.time()
            iter_count = 0
            epoch_losses = []

            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad(set_to_none=True)

                batch_x = batch_x.float().to(self.device, non_blocking=True)
                batch_y = batch_y.float().to(self.device, non_blocking=True)
                batch_x_mark = batch_x_mark.float().to(self.device, non_blocking=True)
                batch_y_mark = batch_y_mark.float().to(self.device, non_blocking=True)

                outputs = self._forward_once(batch_x, batch_y, batch_x_mark, batch_y_mark)
                outputs, target = self._slice_pred_and_true(outputs, batch_y)
                loss = criterion(outputs, target)

                # AMP/FP32 梯度分支
                if self._amp_enabled:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

                epoch_losses.append(float(loss.item()))

                if (i + 1) % 100 == 0:
                    speed = (time.time() - time_now) / iter_count
                    left_iters = (self.args.train_epochs - epoch) * train_steps - (i + 1)
                    left_time = left_iters * speed
                    print(f"\titers: {i + 1}, epoch: {epoch + 1} | loss: {loss.item():.7f}")
                    print(f"\tspeed: {speed:.4f}s/iter; left time: {left_time:.4f}s")
                    iter_count = 0
                    time_now = time.time()

            # 一个 epoch 完成后的度量
            epoch_cost = time.time() - epoch_start
            train_loss = float(np.mean(epoch_losses)) if epoch_losses else float('inf')
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print(f"Epoch: {epoch + 1} cost time: {epoch_cost:.2f}s")
            print(f"Epoch: {epoch + 1}, Steps: {train_steps} | "
                  f"Train Loss: {train_loss:.7f} Vali Loss: {vali_loss:.7f} Test Loss: {test_loss:.7f}")

            early_stopping(vali_loss, self.model, ckpt_dir)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args, train_steps)

        # 载入最佳权重（map 到当前设备）
        best_model_path = os.path.join(ckpt_dir, 'checkpoint.pth')
        self.model.load_state_dict(torch.load(best_model_path, map_location=self.device))
        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(
                torch.load(os.path.join('./checkpoints', setting, 'checkpoint.pth'),
                           map_location=self.device)
            )

        preds, trues = [], []
        folder_path = os.path.join('./test_results', setting)
        os.makedirs(folder_path, exist_ok=True)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device, non_blocking=True)
                batch_y = batch_y.float().to(self.device, non_blocking=True)
                batch_x_mark = batch_x_mark.float().to(self.device, non_blocking=True)
                batch_y_mark = batch_y_mark.float().to(self.device, non_blocking=True)

                outputs = self._forward_once(batch_x, batch_y, batch_x_mark, batch_y_mark)
                # 先统一切片到目标通道
                outputs, target = self._slice_pred_and_true(outputs, batch_y)

                # 回 CPU 做后处理
                outputs_np = outputs.detach().cpu().numpy()
                target_np = target.detach().cpu().numpy()

                # 反归一化（保持与原实现一致）
                if getattr(test_data, "scale", False) and getattr(self.args, "inverse", False):
                    try:
                        shape = target_np.shape
                        if outputs_np.shape[-1] != target_np.shape[-1]:
                            # 通道不一致时，tile 到相同维度（与原版一致）
                            rep = int(target_np.shape[-1] / outputs_np.shape[-1])
                            outputs_np = np.tile(outputs_np, [1, 1, rep])
                        outputs_np = test_data.inverse_transform(outputs_np.reshape(shape[0] * shape[1], -1)).reshape(shape)
                        target_np = test_data.inverse_transform(target_np.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    except Exception as e:
                        print(f"[Warn] inverse_transform failed: {e}")

                preds.append(outputs_np)
                trues.append(target_np)

                # 可视化少量样本（保持原行为，每 20 个 batch 画一次）
                if i % 20 == 0:
                    try:
                        input_np = batch_x.detach().cpu().numpy()
                        if getattr(test_data, "scale", False) and getattr(self.args, "inverse", False):
                            shape_in = input_np.shape
                            input_np = test_data.inverse_transform(
                                input_np.reshape(shape_in[0] * shape_in[1], -1)
                            ).reshape(shape_in)
                        # 取目标通道（最后一维）进行拼接可视化
                        gt = np.concatenate((input_np[0, :, -1], target_np[0, :, -1]), axis=0)
                        pd = np.concatenate((input_np[0, :, -1], outputs_np[0, :, -1]), axis=0)
                        visual(gt, pd, os.path.join(folder_path, f"{i}.pdf"))
                    except Exception as e:
                        print(f"[Warn] visual failed at iter {i}: {e}")

        # 汇总并计算指标
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        print('test shape:', preds.shape, trues.shape)

        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # 结果保存
        metrics_dir = os.path.join('./results', setting)
        os.makedirs(metrics_dir, exist_ok=True)

        # 可选 DTW
        if getattr(self.args, "use_dtw", False):
            dtw_list = []
            manhattan = lambda x, y: np.abs(x - y)
            for i in range(preds.shape[0]):
                if i % 100 == 0:
                    print("calculating dtw iter:", i)
                x = preds[i].reshape(-1, 1)
                y = trues[i].reshape(-1, 1)
                d, _, _, _ = accelerated_dtw(x, y, dist=manhattan)
                dtw_list.append(d)
            dtw_val = float(np.mean(dtw_list)) if dtw_list else float('nan')
        else:
            dtw_val = 'Not calculated'

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print(f'mse:{mse}, mae:{mae}, dtw:{dtw_val}')

        with open("result_long_term_forecast.txt", 'a', encoding='utf-8') as f:
            f.write(setting + "  \n")
            f.write(f'mse:{mse}, mae:{mae}, dtw:{dtw_val}\n\n')

        np.save(os.path.join(metrics_dir, 'metrics.npy'), np.array([mae, mse, rmse, mape, mspe]))
        np.save(os.path.join(metrics_dir, 'pred.npy'), preds)
        np.save(os.path.join(metrics_dir, 'true.npy'), trues)

        return
