import numpy as np
import pandas as pd

# =============================================================================
# 1. الدوال الرياضية المساعدة (THE MATH CORE)
# =============================================================================

def sigmoid(x: np.ndarray) -> np.ndarray:
    """نسخة مستقرة حسابياً من دالة السجمويد لتجنب الـ Overflow."""
    x = np.clip(x, -10, 10)
    return np.where(x >= 0,
                    1.0 / (1.0 + np.exp(-x)),
                    np.exp(x) / (1.0 + np.exp(x)))

def compute_gradients(y: np.ndarray, raw_scores: np.ndarray, 
                      class_weight: float) -> np.ndarray:
    """المشتقة الأولى لـ Binary Cross-Entropy مع تطبيق الوزن."""
    p = sigmoid(raw_scores)
    w = np.where(y == 1, class_weight, 1.0)
    return w * (p - y)

def compute_hessians(y: np.ndarray, raw_scores: np.ndarray, 
                     class_weight: float) -> np.ndarray:
    """المشتقة الثانية (Hessian) مع تطبيق الوزن."""
    p = sigmoid(raw_scores)
    w = np.where(y == 1, class_weight, 1.0)
    return w * p * (1.0 - p)


# =============================================================================
# 2. هيكل الشجرة (TREE STRUCTURE COMPONENTS)
# =============================================================================

class TreeNode:
    """عقدة وحيدة في شجرة القرار."""
    def __init__(self):
        self.is_leaf = False
        self.leaf_value = None  # قيمة التوقع إذا كانت ورقة
        self.feature_idx = None # مؤشر الميزة المستخدمة للتقسيم
        self.threshold = None   # قيمة العتبة للتقسيم
        self.left = None        # الفرع الأيسر (x <= threshold)
        self.right = None       # الفرع الأيمن (x > threshold)


class DecisionTreeXGB:
    """شجرة قرار مخصصة للعمل مع Gradients و Hessians الخاصة بـ XGBoost."""
    def __init__(self, max_depth: int = 3, lam: float = 1.0,
                 gamma: float = 0.0, n_bins: int = 10,
                 min_child_weight: float = 1.0):
        self.max_depth = max_depth
        self.lam = lam
        self.gamma = gamma
        self.n_bins = n_bins
        self.min_child_weight = min_child_weight
        self.root = None

    def _get_split_candidates(self, x: np.ndarray) -> np.ndarray:
        """توليد مرشحي التقسيم بناءً على النسب المئوية (Percentiles)."""
        percentiles = np.linspace(0, 100, self.n_bins + 2)[1:-1]
        return np.unique(np.percentile(x, percentiles))

    @staticmethod
    def _leaf_weight(G: float, H: float, lam: float) -> float:
        """حساب الوزن الأمثل للورقة (FR-006)."""
        return -G / (H + lam)

    @staticmethod
    def _gain(G, H, G_L, H_L, G_R, H_R, lam, gamma):
        """حساب مقدار التحسن (Gain) الناتج عن التقسيم (FR-007)."""
        score_p = G**2 / (H + lam)
        score_l = G_L**2 / (H_L + lam)
        score_r = G_R**2 / (H_R + lam)
        return 0.5 * (score_l + score_r - score_p) - gamma

    def _build(self, X, g, h, depth) -> TreeNode:
        node = TreeNode()
        G, H = g.sum(), h.sum()

        # شروط التوقف: الوصول لأقصى عمق أو عينات غير كافية
        if depth >= self.max_depth or len(g) < 2:
            node.is_leaf = True
            node.leaf_value = self._leaf_weight(G, H, self.lam)
            return node

        best_gain, best_feat, best_thresh, best_left_mask = 0.0, None, None, None
        n_features = X.shape[1]

        for feat in range(n_features):
            candidates = self._get_split_candidates(X[:, feat])
            for thresh in candidates:
                left_mask = X[:, feat] <= thresh
                right_mask = ~left_mask
                if left_mask.sum() == 0 or right_mask.sum() == 0: continue

                G_L, H_L = g[left_mask].sum(), h[left_mask].sum()
                G_R, H_R = g[right_mask].sum(), h[right_mask].sum()

                # رفض التقسيم إذا كان وزن الطفل أقل من المسموح
                if H_L < self.min_child_weight or H_R < self.min_child_weight: continue

                gain = self._gain(G, H, G_L, H_L, G_R, H_R, self.lam, self.gamma)
                if gain > best_gain:
                    best_gain, best_feat, best_thresh, best_left_mask = gain, feat, thresh, left_mask

        if best_feat is None:
            node.is_leaf = True
            node.leaf_value = self._leaf_weight(G, H, self.lam)
            return node

        node.feature_idx, node.threshold = best_feat, best_thresh
        node.left = self._build(X[best_left_mask], g[best_left_mask], h[best_left_mask], depth + 1)
        node.right = self._build(X[~best_left_mask], g[~best_left_mask], h[~best_left_mask], depth + 1)
        return node

    def fit(self, X, g, h):
        self.root = self._build(X, g, h, depth=0)

    def _predict_row(self, row, node) -> float:
        if node.is_leaf: return node.leaf_value
        if row[node.feature_idx] <= node.threshold:
            return self._predict_row(row, node.left)
        return self._predict_row(row, node.right)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.array([self._predict_row(row, self.root) for row in X])


# =============================================================================
# 3. الموديل الرئيسي (XGBOOST FROM SCRATCH)
# =============================================================================

class XGBoostFromScratch:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3, 
                 lam=1.0, gamma=0.0, n_bins=10, min_child_weight=1.0, class_weight=None):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.lam = lam
        self.gamma = gamma
        self.n_bins = n_bins
        self.min_child_weight = min_child_weight
        self._class_weight = class_weight
        self.trees_ = []
        self.loss_history_ = []
        self.stopped_early_ = False

    def fit(self, X, y, verbose=True, log_interval=10, early_stopping_rounds=None):
        """Run the boosting loop.

        Parameters
        ----------
        X, y : training features and binary labels.
        verbose : print round-level progress every ``log_interval`` rounds.
        log_interval : how often (in rounds) to print progress when verbose.
        early_stopping_rounds : stop if training loss does not improve for this
            many consecutive rounds. ``None`` disables early stopping.
        """
        y = np.asarray(y, dtype=np.float64)

        # حساب وزن الكلاس تلقائياً إذا لم يتم تحديده
        if self._class_weight is None:
            n_neg, n_pos = (y == 0).sum(), (y == 1).sum()
            self.class_weight_ = float(np.sqrt(n_neg / max(n_pos, 1)))
        else:
            self.class_weight_ = float(self._class_weight)

        # Reset state so re-fitting works
        self.trees_ = []
        self.loss_history_ = []
        self.stopped_early_ = False

        # البدء بالـ Log-odds للـ base rate
        base_rate = y.mean()
        self._base_score = np.log((base_rate + 1e-9) / (1.0 - base_rate + 1e-9))
        raw_scores = np.full(len(y), self._base_score, dtype=np.float64)

        best_loss = np.inf
        rounds_no_improve = 0

        for t in range(self.n_estimators):
            g = compute_gradients(y, raw_scores, self.class_weight_)
            h = compute_hessians(y, raw_scores, self.class_weight_)

            tree = DecisionTreeXGB(self.max_depth, self.lam, self.gamma,
                                   self.n_bins, self.min_child_weight)
            tree.fit(X, g, h)
            self.trees_.append(tree)

            # تحديث الدرجات الخام (Additive Modeling)
            raw_scores += self.learning_rate * tree.predict(X)

            # Log weighted BCE loss for plotting / early stopping
            p = np.clip(sigmoid(raw_scores), 1e-9, 1.0 - 1e-9)
            w = np.where(y == 1, self.class_weight_, 1.0)
            loss = float(-np.mean(w * (y * np.log(p) + (1.0 - y) * np.log(1.0 - p))))
            self.loss_history_.append(loss)

            if verbose and ((t + 1) % log_interval == 0 or t == 0):
                print(f"  round {t+1:3d}/{self.n_estimators}  loss={loss:.5f}")

            if early_stopping_rounds is not None:
                if loss < best_loss - 1e-6:
                    best_loss = loss
                    rounds_no_improve = 0
                else:
                    rounds_no_improve += 1
                    if rounds_no_improve >= early_stopping_rounds:
                        self.stopped_early_ = True
                        if verbose:
                            print(f"  early stopping at round {t+1} "
                                  f"(no improvement for {early_stopping_rounds} rounds)")
                        break

        return self

    def predict_proba(self, X):
        """توقع الاحتمالات [0, 1]."""
        scores = np.full(len(X), self._base_score, dtype=np.float64)
        for tree in self.trees_:
            scores += self.learning_rate * tree.predict(X)
        return sigmoid(scores)

    def predict(self, X, threshold=0.5):
        """توقع الكلاس بناءً على الـ threshold."""
        return (self.predict_proba(X) >= threshold).astype(int)

    def get_feature_importance(self, feature_names=None):
        """حساب أهمية المتغيرات بناءً على أوزان الأوراق المطلقة."""
        importance_dict = {}
        for tree in self.trees_:
            def traverse(node):
                if node.is_leaf: return
                f_idx = node.feature_idx
                # تقدير الأهمية بناءً على الفرق بين أوزان الأبناء
                val = abs(node.left.leaf_value if node.left.is_leaf else 0) + \
                      abs(node.right.leaf_value if node.right.is_leaf else 0)
                importance_dict[f_idx] = importance_dict.get(f_idx, 0) + val
                traverse(node.left)
                traverse(node.right)
            if tree.root: traverse(tree.root)
        
        features = feature_names if feature_names is not None else [f"V{i}" for i in range(len(importance_dict))]
        data = [{"Feature": features[i], "importance": val} for i, val in importance_dict.items()]
        df = pd.DataFrame(data).sort_values(by="importance", ascending=False)
        total = df["importance"].sum()
        df["% of total"] = (df["importance"] / total * 100).round(1).astype(str) + "%"
        return df