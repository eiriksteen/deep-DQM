import numpy as np
import optuna
from tqdm import tqdm
from sklearn import metrics
from optuna.samplers import TPESampler


class ChiSquareModel:

    def __init__(
            self,
            histograms: np.ndarray,
            is_anomaly: np.ndarray,
            histo_nbins_dict: dict,
            eps: float = 1e-9,
            alpha: float = 0.7,
            EWMA_par1: float = 0.99,
            EWMA_par2: float = 0.99,
            optimise_hyperparameters: int = True,
            score: str = "auc",
            beta: float = 1,
            num_optuna_trials: int = 100
    ):

        self.histograms = histograms
        self.is_anomaly = is_anomaly

        self.eps = eps
        self.alpha = alpha
        self.EWMA_par1 = EWMA_par1
        self.EWMA_par2 = EWMA_par2
        self.optimise_parameters = optimise_hyperparameters
        self.score = score
        self.beta = beta
        self.num_optuna_trials = num_optuna_trials

        self.hist_nbins_dict = histo_nbins_dict
        self.histo_nbins = np.array(list(self.hist_nbins_dict.values()))
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

    def run_algorithm(self, x, y, α):

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
        suma_x0_ = (1 - α) * suma_x_nd + α * suma_x_st

        if suma_x_st != 0 and suma_x_nd != 0:
            x_st = x_st / suma_x_st
            x_nd = x_nd / suma_x_nd
            σz = np.sqrt(x_nd / suma_x_nd - x_nd**2 / suma_x_nd)
            σz[x_nd == 0] = 1 / suma_x_nd
        elif suma_x_st == 0 and suma_x_nd != 0:
            x_nd = x_nd / suma_x_nd
            σz = np.sqrt(x_nd / suma_x_nd - x_nd**2 / suma_x_nd)
            σz[x_nd == 0] = 1 / suma_x_nd
        elif suma_x_st != 0 and suma_x_nd == 0:
            x_st = x_st / suma_x_st
            σz = np.zeros(x_nd.shape[0])
        elif suma_x_st == 0 and suma_x_nd == 0:
            σz = np.zeros(x_nd.shape[0])

        ω = 1 / (σz**2 + self.eps)
        W = (1 - α) * ω
        Sμ = (1 - α) * ω * x_nd
        Sσ = (1 - α) * ω * (x_nd - x_st)**2
        x0_ = Sμ / W
        e0_ = np.sqrt(Sσ / W)

        current_previous_good_run = second_good_run
        usable_run_numbers = []
        χ2s = []
        ndofs = []
        pulls = []
        ground_truth = []
        references = []
        x1s = []
        σ_i_x0 = []
        σ_p_x1 = []

        for i in range(first_usable_run, len(y)):

            if current_previous_good_run != previous_good_run[i]:
                x_upd = x[previous_good_run[i]]
                suma_upd = x_upd.sum()

                if suma_upd != 0:
                    x_upd = x_upd / suma_upd
                    σz = np.sqrt(x_upd / suma_upd - x_upd**2 / suma_upd)
                    σz[x_upd == 0] = 1 / suma_upd
                else:
                    σz = np.zeros(x_nd.shape[0])

                ω = 1 / (σz**2 + self.eps)
                W = α * W + (1 - α) * ω

                Sσ = α * Sσ + (1 - α) * ω * (x_upd - x0_)**2
                e0_ = np.sqrt(Sσ / W)

                Sμ = α * Sμ + (1 - α) * ω * x_upd
                x0_ = Sμ / W

                suma_x0_ = (1 - α) * suma_upd + α * suma_x0_
                current_previous_good_run = previous_good_run[i]

            gt = y[i]
            x1_ = x[i]

            if np.sum(x1_.shape[0]) >= 2:
                ndof = np.sum(x1_.shape[0]) - 1
            elif np.sum(x1_.shape[0]) == 1:
                ndof = 1
            else:
                ndof = 0

            suma_x1_ = x1_.sum()

            if suma_x1_ != 0 and suma_x0_ != 0 and np.abs(suma_x0_) > self.eps and np.abs(suma_x1_) > self.eps:
                x1_ = x1_ / suma_x1_

                σ_x1_ = np.sqrt(x1_ / suma_x1_ - x1_**2 / suma_x1_)
                σ_x0_ = np.sqrt(x0_ / suma_x0_ - x0_**2 / suma_x0_)

                σ_x1_[x1_ == 0] = 1 / suma_x1_
                σ_x0_[x0_ == 0] = 1 / suma_x0_

                if suma_x0_ >= suma_x1_:
                    x0_resampled = np.random.normal(
                        x0_, np.sqrt(e0_**2 + σ_x1_**2))
                    χ2 = np.sum(
                        (x1_ - x0_resampled)**2/(2*σ_x1_**2 + e0_**2))
                    pulls.append(
                        np.sum(
                            (x1_ - x0_resampled)**2/(np.sqrt(2*σ_x1_**2 + e0_**2)))
                    )
                else:
                    x1_resampled = np.random.normal(
                        x1_, np.sqrt(e0_**2 + σ_x0_**2))
                    χ2 = np.sum(
                        (x1_resampled - x0_)**2/(σ_x1_**2 + e0_**2 + σ_x0_**2))
                    pulls.append(
                        np.sum(
                            (x1_resampled - x0_)**2/(np.sqrt(σ_x1_**2 + e0_**2 + σ_x0_**2)))
                    )

            elif suma_x1_ == 0 and suma_x0_ != 0 and np.abs(suma_x0_) > self.eps:
                σ_x1_ = np.zeros(x1_.shape[0])
                σ_x0_ = np.sqrt(x0_ / suma_x0_ - x0_**2 / suma_x0_)
                σ_x0_[x0_ == 0] = 1 / suma_x0_

                x0_resampled = np.random.normal(
                    x0_, np.sqrt(e0_**2 + σ_x1_**2))
                χ2 = np.sum((x1_ - x0_resampled)**2/(2*σ_x1_**2 + e0_**2))
                pulls.append(np.sum((x1_ - x0_resampled)**2 /
                                    (np.sqrt(2*σ_x1_**2 + e0_**2))))

            elif suma_x1_ != 0 and suma_x0_ == 0 and np.abs(suma_x1_) > self.eps:
                x1_ = x1_ / suma_x1_

                σ_x1_ = np.sqrt(x1_ / suma_x1_ - x1_**2 / suma_x1_)
                σ_x0_ = np.zeros(x0_.shape[0])
                σ_x1_[x1_ == 0] = 1 / suma_x1_

                x1_resampled = np.random.normal(
                    x1_, np.sqrt(e0_**2 + σ_x0_**2))
                χ2 = np.sum((x1_resampled - x0_)**2 /
                            (σ_x1_**2 + e0_**2 + σ_x0_**2))
                pulls.append(np.sum((x1_resampled - x0_)**2 /
                                    (np.sqrt(σ_x1_**2 + e0_**2 + σ_x0_**2))))

            else:
                σ_x1_ = np.zeros(x1_.shape[0])
                σ_x0_ = np.zeros(x0_.shape[0])

                χ2 = 0.
                pulls.append(0.)

            # preds_dir = Path("ch2preds")
            # preds_dir.mkdir(exist_ok=True)
            # _, ax = plt.subplots(nrows=2)
            # ax[0].plot(x0_)
            # ax[1].plot(x1_)
            # ax[1].set_title(f"Chi2: {χ2}\nGT: {gt}")
            # plt.savefig(preds_dir / f"run_{i}.png")
            # plt.close()

            ground_truth.append(gt)
            χ2s.append(χ2)
            ndofs.append(ndof)
            usable_run_numbers.append(i)
            references.append(x0_)
            x1s.append(x1_)
            σ_i_x0.append(np.sqrt(e0_**2))  # - σ_x0_**2))
            σ_p_x1.append(σ_x1_)

        return np.array(ground_truth), np.array(χ2s), np.array(ndofs), np.array(usable_run_numbers), np.array(pulls), \
            np.array(references), np.array(
                x1s), np.array(σ_i_x0), np.array(σ_p_x1)

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
            if np.isnan(chisq).any() == False and np.isinf(chisq).any() == False:
                chisq_full += chisq
                ndofs_full += ndofs
                red_chisq_full_sep.append(chisq/ndofs)
                names.append(list(self.hist_nbins_dict.keys())[counter])
                references_all.append(references)
                x1s_all.append(x1s)
                sigma_i_x0_all.append(sigma_i_x0)
                sigma_p_x1_all.append(sigma_p_x1)

            counter += 1

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
