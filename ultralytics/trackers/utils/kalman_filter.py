import numpy as np
import scipy.linalg


class KalmanFilterXYAH:


    def __init__(self):

        ndim, dt = 4, 1.0


        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
        self._update_mat = np.eye(ndim, 2 * ndim)


        self._std_weight_position = 1.0 / 20
        self._std_weight_velocity = 1.0 / 160

    def initiate(self, measurement: np.ndarray) -> tuple:

        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]

        std = [
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[3],
            1e-2,
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            1e-5,
            10 * self._std_weight_velocity * measurement[3],
        ]
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(self, mean: np.ndarray, covariance: np.ndarray) -> tuple:

        std_pos = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-2,
            self._std_weight_position * mean[3],
        ]
        std_vel = [
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[3],
            1e-5,
            self._std_weight_velocity * mean[3],
        ]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))

        mean = np.dot(mean, self._motion_mat.T)
        covariance = np.linalg.multi_dot((self._motion_mat, covariance, self._motion_mat.T)) + motion_cov

        return mean, covariance

    def project(self, mean: np.ndarray, covariance: np.ndarray) -> tuple:

        std = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-1,
            self._std_weight_position * mean[3],
        ]
        innovation_cov = np.diag(np.square(std))

        mean = np.dot(self._update_mat, mean)
        covariance = np.linalg.multi_dot((self._update_mat, covariance, self._update_mat.T))
        return mean, covariance + innovation_cov

    def multi_predict(self, mean: np.ndarray, covariance: np.ndarray) -> tuple:

        std_pos = [
            self._std_weight_position * mean[:, 3],
            self._std_weight_position * mean[:, 3],
            1e-2 * np.ones_like(mean[:, 3]),
            self._std_weight_position * mean[:, 3],
        ]
        std_vel = [
            self._std_weight_velocity * mean[:, 3],
            self._std_weight_velocity * mean[:, 3],
            1e-5 * np.ones_like(mean[:, 3]),
            self._std_weight_velocity * mean[:, 3],
        ]
        sqr = np.square(np.r_[std_pos, std_vel]).T

        motion_cov = [np.diag(sqr[i]) for i in range(len(mean))]
        motion_cov = np.asarray(motion_cov)

        mean = np.dot(mean, self._motion_mat.T)
        left = np.dot(self._motion_mat, covariance).transpose((1, 0, 2))
        covariance = np.dot(left, self._motion_mat.T) + motion_cov

        return mean, covariance

    def update(self, mean: np.ndarray, covariance: np.ndarray, measurement: np.ndarray) -> tuple:

        projected_mean, projected_cov = self.project(mean, covariance)


        projected_cov = (projected_cov + projected_cov.T) / 2
        min_eig = np.min(np.real(np.linalg.eigvalsh(projected_cov)))
        if min_eig < 1e-6:
            projected_cov += np.eye(projected_cov.shape[0]) * (1e-6 - min_eig)

        try:
            chol_factor, lower = scipy.linalg.cho_factor(projected_cov, lower=True, check_finite=False)
            kalman_gain = scipy.linalg.cho_solve(
                (chol_factor, lower), np.dot(covariance, self._update_mat.T).T, check_finite=False
            ).T
        except np.linalg.LinAlgError:

            kalman_gain = np.dot(
                np.dot(covariance, self._update_mat.T),
                np.linalg.pinv(projected_cov)
            )

        innovation = measurement - projected_mean

        new_mean = mean + np.dot(innovation, kalman_gain.T)
        new_covariance = covariance - np.linalg.multi_dot((kalman_gain, projected_cov, kalman_gain.T))

        new_covariance = (new_covariance + new_covariance.T) / 2
        return new_mean, new_covariance

    def gating_distance(
        self,
        mean: np.ndarray,
        covariance: np.ndarray,
        measurements: np.ndarray,
        only_position: bool = False,
        metric: str = "maha",
    ) -> np.ndarray:

        mean, covariance = self.project(mean, covariance)
        if only_position:
            mean, covariance = mean[:2], covariance[:2, :2]
            measurements = measurements[:, :2]

        d = measurements - mean
        if metric == "gaussian":
            return np.sum(d * d, axis=1)
        elif metric == "maha":
            cholesky_factor = np.linalg.cholesky(covariance)
            z = scipy.linalg.solve_triangular(cholesky_factor, d.T, lower=True, check_finite=False, overwrite_b=True)
            return np.sum(z * z, axis=0)
        else:
            raise ValueError("Invalid distance metric")


class KalmanFilterXYWH(KalmanFilterXYAH):


    def initiate(self, measurement: np.ndarray) -> tuple:

        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]

        std = [
            2 * self._std_weight_position * measurement[2],
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[2],
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[2],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[2],
            10 * self._std_weight_velocity * measurement[3],
        ]
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(self, mean, covariance) -> tuple:

        std_pos = [
            self._std_weight_position * mean[2],
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[2],
            self._std_weight_position * mean[3],
        ]
        std_vel = [
            self._std_weight_velocity * mean[2],
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[2],
            self._std_weight_velocity * mean[3],
        ]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))

        mean = np.dot(mean, self._motion_mat.T)
        covariance = np.linalg.multi_dot((self._motion_mat, covariance, self._motion_mat.T)) + motion_cov

        return mean, covariance

    def project(self, mean, covariance) -> tuple:

        std = [
            self._std_weight_position * mean[2],
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[2],
            self._std_weight_position * mean[3],
        ]
        innovation_cov = np.diag(np.square(std))

        mean = np.dot(self._update_mat, mean)
        covariance = np.linalg.multi_dot((self._update_mat, covariance, self._update_mat.T))
        return mean, covariance + innovation_cov

    def multi_predict(self, mean, covariance) -> tuple:

        std_pos = [
            self._std_weight_position * mean[:, 2],
            self._std_weight_position * mean[:, 3],
            self._std_weight_position * mean[:, 2],
            self._std_weight_position * mean[:, 3],
        ]
        std_vel = [
            self._std_weight_velocity * mean[:, 2],
            self._std_weight_velocity * mean[:, 3],
            self._std_weight_velocity * mean[:, 2],
            self._std_weight_velocity * mean[:, 3],
        ]
        sqr = np.square(np.r_[std_pos, std_vel]).T

        motion_cov = [np.diag(sqr[i]) for i in range(len(mean))]
        motion_cov = np.asarray(motion_cov)

        mean = np.dot(mean, self._motion_mat.T)
        left = np.dot(self._motion_mat, covariance).transpose((1, 0, 2))
        covariance = np.dot(left, self._motion_mat.T) + motion_cov

        return mean, covariance

    def update(self, mean, covariance, measurement) -> tuple:

        return super().update(mean, covariance, measurement)


class WormMidpointUKF:


    def __init__(self, dt=1.0):

        self.n = 5
        self.m = 2
        self.dt = dt


        self.alpha = 1e-3
        self.beta = 2.0
        self.kappa = 0.0

        self.lam = self.alpha ** 2 * (self.n + self.kappa) - self.n
        self.gamma = np.sqrt(self.n + self.lam)


        self.Wm = np.full(2 * self.n + 1, 1.0 / (2 * (self.n + self.lam)))
        self.Wc = np.full(2 * self.n + 1, 1.0 / (2 * (self.n + self.lam)))
        self.Wm[0] = self.lam / (self.n + self.lam)
        self.Wc[0] = self.lam / (self.n + self.lam) + (1 - self.alpha ** 2 + self.beta)


        self.Q = np.diag([
            4.0,
            4.0,
            2.0,
            0.3,
            0.1,
        ])


        self.R = np.diag([
            2.0,
            2.0,
        ])

    def initiate(self, measurement):

        x, y = measurement[0], measurement[1]
        mean = np.array([x, y, 0.0, 0.0, 0.0], dtype=np.float64)


        covariance = np.diag([
            1.0,
            1.0,
            10.0,
            np.pi ** 2,
            1.0,
        ])

        return mean, covariance

    def _ctrv_motion(self, state, dt):

        x, y, v, theta, omega = state


        if abs(omega) > 1e-5:
            x_new = x + v / omega * (np.sin(theta + omega * dt) - np.sin(theta))
            y_new = y + v / omega * (np.cos(theta) - np.cos(theta + omega * dt))
        else:

            x_new = x + v * np.cos(theta) * dt
            y_new = y + v * np.sin(theta) * dt

        v_new = v
        theta_new = theta + omega * dt
        omega_new = omega


        theta_new = (theta_new + np.pi) % (2 * np.pi) - np.pi

        return np.array([x_new, y_new, v_new, theta_new, omega_new])

    def _observation_model(self, state):

        return state[:2]

    def _generate_sigma_points(self, mean, covariance):

        n = len(mean)
        sigma_points = np.zeros((2 * n + 1, n))
        sigma_points[0] = mean


        try:

            cov_stabilized = covariance + 1e-8 * np.eye(n)
            sqrt_cov = np.linalg.cholesky((n + self.lam) * cov_stabilized)
        except np.linalg.LinAlgError:

            eigvals, eigvecs = np.linalg.eigh(covariance)
            eigvals = np.maximum(eigvals, 1e-8)
            sqrt_cov = eigvecs @ np.diag(np.sqrt((n + self.lam) * eigvals))

        for i in range(n):
            sigma_points[i + 1] = mean + sqrt_cov[i]
            sigma_points[n + i + 1] = mean - sqrt_cov[i]

        return sigma_points

    def predict(self, mean, covariance):


        sigma_points = self._generate_sigma_points(mean, covariance)


        n_sigma = 2 * self.n + 1
        sigma_pred = np.zeros((n_sigma, self.n))
        for i in range(n_sigma):
            sigma_pred[i] = self._ctrv_motion(sigma_points[i], self.dt)


        pred_mean = np.zeros(self.n)
        for i in range(n_sigma):
            pred_mean += self.Wm[i] * sigma_pred[i]


        pred_mean[3] = (pred_mean[3] + np.pi) % (2 * np.pi) - np.pi


        pred_cov = self.Q.copy()
        for i in range(n_sigma):
            diff = sigma_pred[i] - pred_mean

            diff[3] = (diff[3] + np.pi) % (2 * np.pi) - np.pi
            pred_cov += self.Wc[i] * np.outer(diff, diff)

        return pred_mean, pred_cov

    def update(self, mean, covariance, measurement):


        sigma_points = self._generate_sigma_points(mean, covariance)


        n_sigma = 2 * self.n + 1
        z_sigma = np.zeros((n_sigma, self.m))
        for i in range(n_sigma):
            z_sigma[i] = self._observation_model(sigma_points[i])


        z_mean = np.zeros(self.m)
        for i in range(n_sigma):
            z_mean += self.Wm[i] * z_sigma[i]


        S = self.R.copy()
        Pxz = np.zeros((self.n, self.m))
        for i in range(n_sigma):
            z_diff = z_sigma[i] - z_mean
            x_diff = sigma_points[i] - mean
            x_diff[3] = (x_diff[3] + np.pi) % (2 * np.pi) - np.pi

            S += self.Wc[i] * np.outer(z_diff, z_diff)
            Pxz += self.Wc[i] * np.outer(x_diff, z_diff)


        try:
            K = Pxz @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            K = Pxz @ np.linalg.pinv(S)


        innovation = measurement[:2] - z_mean
        updated_mean = mean + K @ innovation
        updated_cov = covariance - K @ S @ K.T


        updated_mean[3] = (updated_mean[3] + np.pi) % (2 * np.pi) - np.pi

        return updated_mean, updated_cov

    def multi_predict(self, means, covariances):

        N = len(means)
        pred_means = np.zeros_like(means)
        pred_covs = np.zeros_like(covariances)

        for i in range(N):
            pred_means[i], pred_covs[i] = self.predict(means[i], covariances[i])

        return pred_means, pred_covs

    def get_predicted_midpoint(self, mean):

        return mean[:2].copy()
