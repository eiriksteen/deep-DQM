import numpy as np
import pandas as pd
import optuna
from sklearn import metrics
from pathlib import Path
from tqdm import tqdm
from optuna.samplers import TPESampler
from ..settings import HISTO_NBINS_DICT


class ChiSquareModel:

    def __init__(
            self,
            data_path: Path,
            eps: float = 1e-9,
            alpha: float = 0.7,
            EWMA_par1: float = 0.99,
            EWMA_par2: float = 0.99,
            optimise_hyperparameters: int = True,
            score: str = "auc",
            beta: float = 1,
            num_optuna_trials: int = 100
    ):

        self.eps = eps
        self.alpha = alpha
        self.EWMA_par1 = EWMA_par1
        self.EWMA_par2 = EWMA_par2
        self.optimise_parameters = optimise_hyperparameters
        self.score = score
        self.beta = beta
        self.num_optuna_trials = num_optuna_trials

        df = pd.read_csv(data_path, header=0)
        input_var_cols = [
            c for c in df.columns if 'var' in c and not 'err' in c]
        self.histograms = df[input_var_cols].to_numpy()
        self.is_anomaly = 1 - df['all_OK'].to_numpy()

        self.histo_nbins = np.array(list(HISTO_NBINS_DICT.values()))
        self.histo_nbins = self.histo_nbins[np.where(self.histo_nbins != 0)[0]]
        self.n_bins = self.histograms.shape[-1]

    def calculate_metrics(self, Y_val, Y_pred_probs, T_values, n_samples, n_trials, beta):
        eps_good_list = []
        eps_bad_list = []
        eps_balanced_weighed_list = []

        for _ in range(n_trials):
            indx = np.random.choice(
                Y_val.shape[0], size=n_samples, replace=True)
            Y_pred_probs_sample = Y_pred_probs[indx]

            Y_preds = (Y_pred_probs_sample[:, np.newaxis] > T_values).astype(
                np.int32)
            true_positives = np.sum(
                (Y_val[indx][:, np.newaxis] == 1) & (Y_preds == 1), axis=0)
            true_negatives = np.sum(
                (Y_val[indx][:, np.newaxis] == 0) & (Y_preds == 0), axis=0)
            false_positives = np.sum(
                (Y_val[indx][:, np.newaxis] == 0) & (Y_preds == 1), axis=0)
            false_negatives = np.sum(
                (Y_val[indx][:, np.newaxis] == 1) & (Y_preds == 0), axis=0)

            eps_bad = true_positives / (true_positives + false_negatives)
            eps_good = true_negatives / (true_negatives + false_positives)
            eps_balanced_weighed = (eps_good + beta * eps_bad) / (1 + beta)

            eps_good_list.append(eps_good[:-1])
            eps_bad_list.append(eps_bad[:-1])
            eps_balanced_weighed_list.append(eps_balanced_weighed[:-1])

        return eps_good_list, eps_bad_list, eps_balanced_weighed_list

    def run_algorithm(self, x: np.ndarray, y: np.ndarray, alpha: float):

        first_good_run = -1
        second_good_run = -1
        for i in range(len(y)):
            if y[i] == False:
                if first_good_run < 0:
                    first_good_run = i
                elif second_good_run < 0:
                    second_good_run = i
            if second_good_run > 0:
                break

        previous_good_run = [-1]
        for i in range(1, len(y)):
            if i > first_good_run:
                for j in range(i):
                    if y[i-j-1] == False:
                        previous_good_run.append(i-j-1)
                        break
            else:
                previous_good_run.append(-1)

        first_usable_run = previous_good_run.index(second_good_run)

        x_st = x[first_good_run]
        x_nd = x[second_good_run]

        suma_x_st = x_st.sum()
        suma_x_nd = x_nd.sum()
        suma_x0_ = (1 - self.alpha) * suma_x_nd + self.alpha * suma_x_st

        x_st = x_st / suma_x_st
        x_nd = x_nd / suma_x_nd

        sigma_z = np.sqrt(x_nd / suma_x_nd - x_nd**2 / suma_x_nd)
        sigma_z[x_nd == 0] = 1 / suma_x_nd

        ω = 1 / (sigma_z**2 + self.eps)
        W = (1 - self.alpha) * ω
        S_mu = (1 - self.alpha) * ω * x_nd
        S_sigma = (1 - self.alpha) * ω * (x_nd - x_st)**2
        x0_ = S_mu / W
        e0_ = np.sqrt(S_sigma / W)

        current_previous_good_run = second_good_run
        usable_run_numbers = []
        χ2s = []
        ndofs = []
        pulls = []
        ground_truth = []
        references = []
        x1s = []
        sigma_i_x0 = []
        sigma_p_x1 = []

        for i in range(first_usable_run, len(y)):

            if current_previous_good_run != previous_good_run[i]:
                x_upd = x[previous_good_run[i]]
                suma_upd = x_upd.sum()
                x_upd = x_upd / suma_upd

                sigma_z = np.sqrt(x_upd / suma_upd - x_upd**2 / suma_upd)
                sigma_z[x_upd == 0] = 1 / suma_upd

                ω = 1 / (sigma_z**2 + self.eps)
                W = self.alpha * W + (1 - self.alpha) * ω

                S_sigma = self.alpha * S_sigma + \
                    (1 - self.alpha) * ω * (x_upd - x0_)**2
                e0_ = np.sqrt(S_sigma / W)

                S_mu = self.alpha * S_mu + (1 - self.alpha) * ω * x_upd
                x0_ = S_mu / W

                suma_x0_ = (1 - self.alpha) * suma_upd + self.alpha * suma_x0_
                current_previous_good_run = previous_good_run[i]

            gt = y[i]
            x1_ = x[i]

            filter = np.logical_and(e0_ > 0, e0_ != np.inf)
            ndof = np.sum(filter) - 1

            x1_non_zero = x1_[filter]
            x0_non_zero = x0_[filter]
            e0_non_zero = e0_[filter]

            suma_x1_non_zero = x1_non_zero.sum()
            x1_non_zero = x1_non_zero / suma_x1_non_zero

            sigma_x1_non_zero = np.sqrt(
                x1_non_zero / suma_x1_non_zero - x1_non_zero**2 / suma_x1_non_zero)
            sigma_x0_non_zero = np.sqrt(
                x0_non_zero / suma_x0_ - x0_non_zero**2 / suma_x0_)

            sigma_x1_non_zero[x1_non_zero == 0] = 1 / suma_x1_non_zero
            sigma_x0_non_zero[x0_non_zero == 0] = 1 / suma_x0_

            sigma_x1_ = np.sqrt(x1_) / x1_.sum()
            sigma_x0_ = np.sqrt(x0_) / suma_x0_

            sigma_x1_[x1_ == 0] = 1 / suma_x1_non_zero
            sigma_x0_[x0_ == 0] = 1 / suma_x0_

            if suma_x0_ >= suma_x1_non_zero:
                x0_non_zero_resampled = np.random.normal(
                    x0_non_zero, np.sqrt(e0_non_zero**2 + sigma_x1_non_zero**2))
                χ2 = np.sum(
                    (x1_non_zero - x0_non_zero_resampled)**2/(2*sigma_x1_non_zero**2 + e0_non_zero**2))
                pulls.append(
                    np.sum(
                        (x1_non_zero - x0_non_zero_resampled)**2/(np.sqrt(2*sigma_x1_non_zero**2 + e0_non_zero**2)))
                )
            else:
                x1_resampled_non_zero = np.random.normal(
                    x1_non_zero, np.sqrt(e0_non_zero**2 + sigma_x0_non_zero**2))
                χ2 = np.sum(
                    (x1_resampled_non_zero - x0_non_zero)**2/(sigma_x1_non_zero**2 + e0_non_zero**2 + sigma_x0_non_zero**2))
                pulls.append(
                    np.sum(
                        (x1_resampled_non_zero - x0_non_zero)**2/(np.sqrt(sigma_x1_non_zero**2 + e0_non_zero**2 + sigma_x0_non_zero**2)))
                )

            ground_truth.append(gt)
            χ2s.append(χ2)
            ndofs.append(ndof)
            usable_run_numbers.append(i)
            references.append(x0_)
            x1s.append(x1_)
            sigma_i_x0.append(np.sqrt(e0_**2))
            sigma_p_x1.append(sigma_x1_)

        out_arrs = [ground_truth, χ2s, ndofs, usable_run_numbers,
                    pulls, references, x1s, sigma_i_x0, sigma_p_x1]

        return (np.array(arr) for arr in out_arrs)

    def compute_score(self, trial):

        alpha = trial.suggest_float("alpha", 0.0, 1.0)
        chisq_full = 0
        ndofs_full = 0
        index = 0

        for size in self.histo_nbins:
            out = self.run_algorithm(
                self.histograms[:, index:index + size], self.is_anomaly, alpha)
            ground_truth, chisq, ndofs = out[:3]
            index += size
            if np.isnan(chisq).any() == False and np.isinf(chisq).any() == False:
                chisq_full += chisq
                ndofs_full += ndofs

        log_red_chisq = np.log(chisq_full / ndofs_full)
        if self.score == "auc":
            fpr, tpr, _ = metrics.roc_curve(ground_truth, log_red_chisq)
            out_score = np.abs(metrics.auc(fpr, tpr) - 0.5) + 0.5

        elif self.score == "wba":
            xllim, xrlim = np.percentile(
                log_red_chisq, 5), np.percentile(log_red_chisq, 95)
            T_values = np.linspace(xllim, xrlim, 1000)
            n_samples = len(log_red_chisq)
            n_trials = 200
            _, _, eps_balanced_weighed_list = self.calculate_metrics(
                ground_truth, log_red_chisq, T_values, n_samples, n_trials, self.beta)

            median_eps_balanced_weighed = np.quantile(
                eps_balanced_weighed_list, [0.5], axis=0)
            out_score = np.max(median_eps_balanced_weighed)

        return out_score

    def optimise_hyperparameters(self):

        sampler = TPESampler(seed=1)
        study = optuna.create_study(
            study_name="EWMA", direction="maximize", sampler=sampler)
        study.optimize(self.compute_score, n_trials=self.num_optuna_trials)

        print("Number of finished trials: ", len(study.trials))
        print("Best trial:")
        trial = study.best_trial
        print("  Value: ", trial.value)
        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))

        self.alpha = trial.params["alpha"]

    def fit(self):

        if self.optimise_parameters:
            self.optimise_hyperparameters()

        chisq_full = 0
        red_chisq_full_sep = []
        ndofs_full = 0
        index = 0
        counter = 0
        names = []
        references_all = []
        x1s_all = []
        sigma_i_x0_all = []
        sigma_p_x1_all = []

        self.is_anomaly = self.is_anomaly

        for size in tqdm(self.histo_nbins):
            ground_truth, chisq, ndofs, _, _, references, x1s, sigma_i_x0, sigma_p_x1 = self.run_algorithm(
                self.histograms[:, index:index + size], self.is_anomaly, self.alpha)
            index += size
            counter += 1
            if np.isnan(chisq).any() == False and np.isinf(chisq).any() == False:
                chisq_full += chisq
                ndofs_full += ndofs
                red_chisq_full_sep.append(chisq/ndofs)
                names.append(list(HISTO_NBINS_DICT.keys())[counter])
                references_all.append(references)
                x1s_all.append(x1s)
                sigma_i_x0_all.append(sigma_i_x0)
                sigma_p_x1_all.append(sigma_p_x1)

        log_red_chisq = np.log(chisq_full / ndofs_full)
        red_chisq_full_sep = np.array(red_chisq_full_sep).T
        names = np.array(names)

        xllim, xrlim = np.percentile(
            log_red_chisq, 5), np.percentile(log_red_chisq, 95)

        T_values = np.linspace(xllim, xrlim, 1000)

        n_samples = len(log_red_chisq)
        n_trials = 1000

        _, _, eps_balanced_weighed_list = self.calculate_metrics(
            ground_truth, log_red_chisq, T_values, n_samples, n_trials, self.beta)

        _, median_eps_balanced_weighed, _ = np.quantile(
            eps_balanced_weighed_list, [0.5-0.34, 0.5, 0.5+0.34], axis=0)

        T_value_best = T_values[np.argmax(median_eps_balanced_weighed)]

        preds = log_red_chisq > T_value_best

        return log_red_chisq, preds, ground_truth
