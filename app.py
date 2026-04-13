import streamlit as st
import numpy as np
from scipy import stats
from scipy.integrate import quad
import sympy as sp
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

st.sidebar.title("🚀 ML Math Toolkit")

st.set_page_config(page_title="ML Math Toolkit", page_icon="🧮", layout="wide")

st.title("🧮 ML Math Toolkit", anchor=False)
st.caption("Complete Math & AI/ML Calculator — Linear Algebra, Calculus, Probability, ML Functions")
st.divider()

category = st.selectbox("📂 Choose Category",
                        ["Basic Arithmetic",
                         "Statistics",
                         "Probability",
                         "Linear Algebra (Matrix)",
                         "Calculus",
                         "ML Functions (Loss & Activation)"],
                        index=None, placeholder="Select a category...")

st.divider()

# ─────────────────────────────────────────────────────────
# HELPER: Safe math evaluator using sympy
# ─────────────────────────────────────────────────────────
def safe_eval(expr_str, var_val, var_name='x'):
    """Safely evaluate a math expression using sympy."""
    x = sp.Symbol(var_name)
    allowed = {var_name: x, 'sin': sp.sin, 'cos': sp.cos, 'tan': sp.tan,
               'exp': sp.exp, 'log': sp.log, 'sqrt': sp.sqrt,
               'pi': sp.pi, 'e': sp.E, 'abs': sp.Abs}
    expr = sp.sympify(expr_str, locals=allowed)
    return float(expr.subs(x, var_val))

def safe_sympy(expr_str, var_name='x'):
    """Return sympy expression safely."""
    x = sp.Symbol(var_name)
    allowed = {var_name: x, 'sin': sp.sin, 'cos': sp.cos, 'tan': sp.tan,
               'exp': sp.exp, 'log': sp.log, 'sqrt': sp.sqrt,
               'pi': sp.pi, 'e': sp.E, 'abs': sp.Abs}
    return sp.sympify(expr_str, locals=allowed), x

def plot_function(func, x_range=(-5, 5), title="f(x)", label="f(x)"):
    """Plot a function over a range."""
    xs = np.linspace(x_range[0], x_range[1], 400)
    ys = [func(xi) for xi in xs]
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(xs, ys, color='#4f8ef7', linewidth=2, label=label)
    ax.axhline(0, color='gray', linewidth=0.7, linestyle='--')
    ax.axvline(0, color='gray', linewidth=0.7, linestyle='--')
    ax.set_title(title, fontsize=13)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig

def parse_matrix(text):
    return np.array([[float(x) for x in row.split(',')] for row in text.strip().split(';')])

# ─────────────────────────────────────────────────────────
# 1. BASIC ARITHMETIC
# ─────────────────────────────────────────────────────────
if category == "Basic Arithmetic":
    col1, col2 = st.columns(2)
    with col1:
        num1 = st.number_input("1st number", value=None, placeholder="enter number")
    with col2:
        num2 = st.number_input("2nd number", value=None, placeholder="enter number")

    operation = st.selectbox("Operation",
                             ['+', '-', '*', '/', '** (power)', '% (modulo)'],
                             index=None, placeholder="choose operation")

    if st.button("Calculate", type="primary"):
        if num1 is not None and num2 is not None and operation:
            ans = None
            if operation == '+':              ans = num1 + num2
            elif operation == '-':            ans = num1 - num2
            elif operation == '*':            ans = num1 * num2
            elif operation == '/':
                if num2 != 0:                 ans = num1 / num2
                else:                         st.error("❌ Cannot divide by zero!")
            elif operation == '** (power)':   ans = num1 ** num2
            elif operation == '% (modulo)':   ans = num1 % num2
            if ans is not None:
                st.success(f"✅ Answer: **{ans}**")
        else:
            st.warning("⚠️ Fill in all fields!")

# ─────────────────────────────────────────────────────────
# 2. STATISTICS
# ─────────────────────────────────────────────────────────
elif category == "Statistics":
    st.info("💡 Enter comma-separated numbers. Example: `1, 2, 3, 4, 5`")
    data_input = st.text_input("Enter your data", placeholder="e.g. 2, 4, 6, 8, 10")
    operation = st.selectbox("Operation",
                             ["Mean", "Median", "Mode", "Variance",
                              "Standard Deviation", "Min", "Max", "Range",
                              "Percentile", "IQR", "Skewness", "Kurtosis"],
                             index=None, placeholder="choose operation")
    percentile_val = 50
    if operation == "Percentile":
        percentile_val = st.number_input("Percentile (0-100)", min_value=0, max_value=100, value=50)

    if st.button("Calculate", type="primary"):
        if data_input and operation:
            try:
                data = [float(x.strip()) for x in data_input.split(',')]
                arr = np.array(data)
                result = None
                if operation == "Mean":                  result = np.mean(arr)
                elif operation == "Median":              result = np.median(arr)
                elif operation == "Mode":                result = stats.mode(arr, keepdims=True).mode[0]
                elif operation == "Variance":            result = np.var(arr)
                elif operation == "Standard Deviation":  result = np.std(arr)
                elif operation == "Min":                 result = np.min(arr)
                elif operation == "Max":                 result = np.max(arr)
                elif operation == "Range":               result = np.max(arr) - np.min(arr)
                elif operation == "Percentile":          result = np.percentile(arr, percentile_val)
                elif operation == "IQR":                 result = stats.iqr(arr)
                elif operation == "Skewness":            result = stats.skew(arr)
                elif operation == "Kurtosis":            result = stats.kurtosis(arr)
                st.success(f"✅ {operation}: **{result:.6f}**")

                # Distribution plot
                fig, ax = plt.subplots(figsize=(7, 3))
                ax.hist(arr, bins='auto', color='#4f8ef7', edgecolor='white', alpha=0.8)
                ax.axvline(np.mean(arr), color='red', linestyle='--', label=f'Mean={np.mean(arr):.2f}')
                ax.set_title("Data Distribution", fontsize=13)
                ax.legend(); ax.grid(True, alpha=0.3)
                fig.tight_layout()
                st.pyplot(fig)
            except Exception as e:
                st.error(f"❌ Error: {e}")
        else:
            st.warning("⚠️ Fill in all fields!")

# ─────────────────────────────────────────────────────────
# 3. PROBABILITY
# ─────────────────────────────────────────────────────────
elif category == "Probability":
    operation = st.selectbox("Operation",
                             ["Basic Probability (favourable/total)",
                              "Complement P(A')",
                              "Union P(A∪B)",
                              "Intersection P(A∩B) — Independent",
                              "Conditional Probability P(A|B)",
                              "Bayes Theorem",
                              "Binomial Distribution",
                              "Poisson Distribution",
                              "Exponential Distribution",
                              "Gaussian PDF",
                              "Normal Distribution Z-score"],
                             index=None, placeholder="choose operation")

    if operation and st.button("Calculate", type="primary"):

        if operation == "Basic Probability (favourable/total)":
            fav   = st.number_input("Favourable outcomes", min_value=0, value=3)
            total = st.number_input("Total outcomes", min_value=1, value=10)
            st.success(f"P = {fav}/{total} = **{fav/total:.4f}**")

        elif operation == "Complement P(A')":
            p = st.number_input("P(A)", min_value=0.0, max_value=1.0, value=0.4)
            st.success(f"P(A') = 1 - {p} = **{1-p:.4f}**")

        elif operation == "Union P(A∪B)":
            pa  = st.number_input("P(A)", min_value=0.0, max_value=1.0, value=0.5)
            pb  = st.number_input("P(B)", min_value=0.0, max_value=1.0, value=0.3)
            pab = st.number_input("P(A∩B)", min_value=0.0, max_value=1.0, value=0.1)
            st.success(f"P(A∪B) = **{pa + pb - pab:.4f}**")

        elif operation == "Intersection P(A∩B) — Independent":
            pa = st.number_input("P(A)", min_value=0.0, max_value=1.0, value=0.5)
            pb = st.number_input("P(B)", min_value=0.0, max_value=1.0, value=0.4)
            st.success(f"P(A∩B) = **{pa * pb:.4f}**")

        elif operation == "Conditional Probability P(A|B)":
            st.latex(r"P(A|B) = \frac{P(A \cap B)}{P(B)}")
            pab = st.number_input("P(A∩B)", min_value=0.0, max_value=1.0, value=0.2)
            pb  = st.number_input("P(B)",   min_value=0.001, max_value=1.0, value=0.5)
            st.success(f"P(A|B) = **{pab/pb:.4f}**")

        elif operation == "Bayes Theorem":
            st.latex(r"P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)}")
            pba = st.number_input("P(B|A)", min_value=0.0, max_value=1.0, value=0.9)
            pa  = st.number_input("P(A)",   min_value=0.0, max_value=1.0, value=0.3)
            pb  = st.number_input("P(B)",   min_value=0.001, max_value=1.0, value=0.4)
            st.success(f"P(A|B) = **{(pba * pa) / pb:.4f}**")

        elif operation == "Binomial Distribution":
            st.latex(r"P(X=k) = \binom{n}{k} p^k (1-p)^{n-k}")
            n = int(st.number_input("n (trials)", min_value=1, value=10))
            k = int(st.number_input("k (successes)", min_value=0, value=3))
            p = st.number_input("p (success prob)", min_value=0.0, max_value=1.0, value=0.5)
            result = stats.binom.pmf(k, n, p)
            st.success(f"P(X={k}) = **{result:.6f}**")
            xs = np.arange(0, n+1)
            fig, ax = plt.subplots(figsize=(7, 3))
            ax.bar(xs, stats.binom.pmf(xs, n, p), color='#4f8ef7', edgecolor='white')
            ax.axvline(k, color='red', linestyle='--', label=f'k={k}')
            ax.set_title("Binomial Distribution PMF"); ax.legend(); ax.grid(True, alpha=0.3)
            st.pyplot(fig)

        elif operation == "Poisson Distribution":
            st.latex(r"P(X=k) = \frac{\mu^k e^{-\mu}}{k!}")
            mu = st.number_input("μ (average rate)", min_value=0.1, value=3.0)
            k  = int(st.number_input("k", min_value=0, value=2))
            result = stats.poisson.pmf(k, mu)
            st.success(f"P(X={k}) = **{result:.6f}**")
            xs = np.arange(0, int(mu*3)+1)
            fig, ax = plt.subplots(figsize=(7, 3))
            ax.bar(xs, stats.poisson.pmf(xs, mu), color='#4f8ef7', edgecolor='white')
            ax.axvline(k, color='red', linestyle='--', label=f'k={k}')
            ax.set_title("Poisson Distribution PMF"); ax.legend(); ax.grid(True, alpha=0.3)
            st.pyplot(fig)

        elif operation == "Exponential Distribution":
            st.latex(r"f(x) = \lambda e^{-\lambda x}")
            lam = st.number_input("λ (rate)", min_value=0.01, value=1.0)
            x   = st.number_input("x", min_value=0.0, value=1.0)
            pdf = stats.expon.pdf(x, scale=1/lam)
            st.success(f"f({x}) = **{pdf:.6f}**")
            xs = np.linspace(0, 5/lam, 300)
            fig, ax = plt.subplots(figsize=(7, 3))
            ax.plot(xs, stats.expon.pdf(xs, scale=1/lam), color='#4f8ef7', linewidth=2)
            ax.axvline(x, color='red', linestyle='--', label=f'x={x}')
            ax.set_title("Exponential Distribution PDF"); ax.legend(); ax.grid(True, alpha=0.3)
            st.pyplot(fig)

        elif operation == "Gaussian PDF":
            st.latex(r"f(x) = \frac{1}{\sigma\sqrt{2\pi}} e^{-\frac{(x-\mu)^2}{2\sigma^2}}")
            mu    = st.number_input("μ (mean)", value=0.0)
            sigma = st.number_input("σ (std dev)", min_value=0.01, value=1.0)
            x     = st.number_input("x", value=0.0)
            pdf   = stats.norm.pdf(x, mu, sigma)
            st.success(f"f({x}) = **{pdf:.6f}**")
            xs = np.linspace(mu - 4*sigma, mu + 4*sigma, 300)
            fig, ax = plt.subplots(figsize=(7, 3))
            ax.plot(xs, stats.norm.pdf(xs, mu, sigma), color='#4f8ef7', linewidth=2)
            ax.axvline(x, color='red', linestyle='--', label=f'x={x}')
            ax.fill_between(xs, stats.norm.pdf(xs, mu, sigma), alpha=0.15, color='#4f8ef7')
            ax.set_title("Gaussian (Normal) PDF"); ax.legend(); ax.grid(True, alpha=0.3)
            st.pyplot(fig)

        elif operation == "Normal Distribution Z-score":
            st.latex(r"Z = \frac{X - \mu}{\sigma}")
            x     = st.number_input("X (value)", value=70.0)
            mu    = st.number_input("μ (mean)", value=60.0)
            sigma = st.number_input("σ (std dev)", min_value=0.01, value=10.0)
            z = (x - mu) / sigma
            p = stats.norm.cdf(z)
            st.success(f"Z = **{z:.4f}** | P(X ≤ {x}) = **{p:.4f}**")

# ─────────────────────────────────────────────────────────
# 4. LINEAR ALGEBRA
# ─────────────────────────────────────────────────────────
elif category == "Linear Algebra (Matrix)":
    st.info("💡 Enter rows separated by `;`, values by `,`  →  Example: `1,2;3,4`")

    operation = st.selectbox("Operation",
                             ["Addition", "Subtraction", "Multiplication",
                              "Transpose", "Determinant", "Inverse",
                              "Rank", "Trace", "Norm (L1 & L2)",
                              "Eigenvalues & Eigenvectors",
                              "SVD (Singular Value Decomposition)",
                              "Dot Product"],
                             index=None, placeholder="choose operation")

    single_ops = ["Transpose", "Determinant", "Inverse", "Rank",
                  "Trace", "Norm (L1 & L2)", "Eigenvalues & Eigenvectors",
                  "SVD (Singular Value Decomposition)"]

    m1_input = st.text_input("Matrix A", placeholder="e.g. 1,2;3,4")
    m2_input = None
    if operation and operation not in single_ops:
        m2_input = st.text_input("Matrix B", placeholder="e.g. 5,6;7,8")

    if st.button("Calculate", type="primary"):
        if operation and m1_input:
            try:
                A = parse_matrix(m1_input)

                if operation == "Transpose":
                    st.success("Transpose A^T:"); st.dataframe(A.T)

                elif operation == "Determinant":
                    st.success(f"det(A) = **{np.linalg.det(A):.4f}**")

                elif operation == "Inverse":
                    st.success("A⁻¹:"); st.dataframe(np.linalg.inv(A))

                elif operation == "Rank":
                    st.success(f"Rank of A = **{np.linalg.matrix_rank(A)}**")

                elif operation == "Trace":
                    st.success(f"Trace of A = **{np.trace(A):.4f}**")

                elif operation == "Norm (L1 & L2)":
                    l1 = np.linalg.norm(A, ord=1)
                    l2 = np.linalg.norm(A, ord=2)
                    fn = np.linalg.norm(A, 'fro')
                    col1, col2, col3 = st.columns(3)
                    col1.metric("L1 Norm", f"{l1:.4f}")
                    col2.metric("L2 (Spectral) Norm", f"{l2:.4f}")
                    col3.metric("Frobenius Norm", f"{fn:.4f}")

                elif operation == "Eigenvalues & Eigenvectors":
                    vals, vecs = np.linalg.eig(A)
                    st.success(f"Eigenvalues: **{vals}**")
                    st.write("Eigenvectors (columns):"); st.dataframe(vecs)

                elif operation == "SVD (Singular Value Decomposition)":
                    U, S, Vt = np.linalg.svd(A)
                    st.success("SVD: A = U · Σ · Vᵀ")
                    col1, col2, col3 = st.columns(3)
                    with col1: st.write("**U:**"); st.dataframe(U)
                    with col2: st.write("**Σ (singular values):**"); st.write(S)
                    with col3: st.write("**Vᵀ:**"); st.dataframe(Vt)

                elif m2_input:
                    B = parse_matrix(m2_input)
                    if operation == "Addition":
                        st.success("A + B:"); st.dataframe(A + B)
                    elif operation == "Subtraction":
                        st.success("A - B:"); st.dataframe(A - B)
                    elif operation == "Multiplication":
                        st.success("A @ B:"); st.dataframe(A @ B)
                    elif operation == "Dot Product":
                        st.success(f"Dot Product = **{np.dot(A.flatten(), B.flatten()):.4f}**")
            except Exception as e:
                st.error(f"❌ Error: {e}")
        else:
            st.warning("⚠️ Fill in all fields!")

# ─────────────────────────────────────────────────────────
# 5. CALCULUS (with sympy — safe, no eval)
# ─────────────────────────────────────────────────────────
elif category == "Calculus":
    st.info("💡 Supported: `x**2`, `sin(x)`, `cos(x)`, `exp(x)`, `log(x)`, `sqrt(x)`, `abs(x)`")

    operation = st.selectbox("Operation",
                             ["Derivative (dy/dx)",
                              "Second Derivative (d²y/dx²)",
                              "Partial Derivative (∂f/∂x or ∂f/∂y)",
                              "Gradient Vector ∇f",
                              "Definite Integral",
                              "Gradient Descent",
                              "Straight Line (y = mx + c)"],
                             index=None, placeholder="choose operation")

    if operation == "Derivative (dy/dx)":
        st.latex(r"\frac{d}{dx} f(x)")
        func_input = st.text_input("f(x)", placeholder="e.g. x**3 + 2*x")
        x_val = st.number_input("At x =", value=2.0)
        if st.button("Calculate", type="primary"):
            try:
                expr, x = safe_sympy(func_input)
                deriv = sp.diff(expr, x)
                result = float(deriv.subs(x, x_val))
                st.success(f"f'(x) = `{deriv}`")
                st.success(f"f'({x_val}) = **{result:.6f}**")
                st.pyplot(plot_function(lambda xi: safe_eval(func_input, xi),
                                        title="f(x) and its Derivative"))
            except Exception as e:
                st.error(f"❌ Error: {e}")

    elif operation == "Second Derivative (d²y/dx²)":
        st.latex(r"\frac{d^2}{dx^2} f(x)")
        func_input = st.text_input("f(x)", placeholder="e.g. x**4 - 3*x**2")
        x_val = st.number_input("At x =", value=1.0)
        if st.button("Calculate", type="primary"):
            try:
                expr, x = safe_sympy(func_input)
                d2 = sp.diff(expr, x, 2)
                result = float(d2.subs(x, x_val))
                st.success(f"f''(x) = `{d2}`")
                st.success(f"f''({x_val}) = **{result:.6f}**")
                concavity = "Concave Up (local min candidate) 📈" if result > 0 else "Concave Down (local max candidate) 📉"
                st.info(f"At x={x_val}: {concavity}")
            except Exception as e:
                st.error(f"❌ Error: {e}")

    elif operation == "Partial Derivative (∂f/∂x or ∂f/∂y)":
        st.latex(r"\frac{\partial f}{\partial x}, \quad \frac{\partial f}{\partial y}")
        st.info("💡 Use `x` and `y` in your function. Example: `x**2 + 3*x*y + y**2`")
        func_input = st.text_input("f(x, y)", placeholder="e.g. x**2 + 3*x*y + y**2")
        wrt = st.selectbox("Differentiate with respect to", ["x", "y"])
        x_val = st.number_input("x =", value=1.0)
        y_val = st.number_input("y =", value=2.0)
        if st.button("Calculate", type="primary"):
            try:
                x, y = sp.symbols('x y')
                allowed = {'x': x, 'y': y, 'sin': sp.sin, 'cos': sp.cos,
                           'exp': sp.exp, 'log': sp.log, 'sqrt': sp.sqrt}
                expr = sp.sympify(func_input, locals=allowed)
                var = x if wrt == 'x' else y
                partial = sp.diff(expr, var)
                result = float(partial.subs([(x, x_val), (y, y_val)]))
                st.success(f"∂f/∂{wrt} = `{partial}`")
                st.success(f"At ({x_val}, {y_val}): **{result:.6f}**")
            except Exception as e:
                st.error(f"❌ Error: {e}")

    elif operation == "Gradient Vector ∇f":
        st.latex(r"\nabla f = \left[\frac{\partial f}{\partial x}, \frac{\partial f}{\partial y}\right]")
        st.info("💡 Use `x` and `y`. Example: `x**2 + y**2`")
        func_input = st.text_input("f(x, y)", placeholder="e.g. x**2 + y**2")
        x_val = st.number_input("x =", value=1.0)
        y_val = st.number_input("y =", value=2.0)
        if st.button("Calculate", type="primary"):
            try:
                x, y = sp.symbols('x y')
                allowed = {'x': x, 'y': y, 'sin': sp.sin, 'cos': sp.cos,
                           'exp': sp.exp, 'log': sp.log, 'sqrt': sp.sqrt}
                expr = sp.sympify(func_input, locals=allowed)
                gx = sp.diff(expr, x)
                gy = sp.diff(expr, y)
                gx_val = float(gx.subs([(x, x_val), (y, y_val)]))
                gy_val = float(gy.subs([(x, x_val), (y, y_val)]))
                magnitude = np.sqrt(gx_val**2 + gy_val**2)
                st.success(f"∇f = [`{gx}`, `{gy}`]")
                col1, col2, col3 = st.columns(3)
                col1.metric("∂f/∂x", f"{gx_val:.4f}")
                col2.metric("∂f/∂y", f"{gy_val:.4f}")
                col3.metric("|∇f| (magnitude)", f"{magnitude:.4f}")
            except Exception as e:
                st.error(f"❌ Error: {e}")

    elif operation == "Definite Integral":
        st.latex(r"\int_a^b f(x)\,dx")
        func_input = st.text_input("f(x)", placeholder="e.g. x**2")
        col1, col2 = st.columns(2)
        a = col1.number_input("Lower limit (a)", value=0.0)
        b = col2.number_input("Upper limit (b)", value=1.0)
        if st.button("Calculate", type="primary"):
            try:
                f = lambda xi: safe_eval(func_input, xi)
                result, error = quad(f, a, b)
                st.success(f"∫f(x)dx from {a} to {b} = **{result:.6f}**")
                st.caption(f"Estimated error: {error:.2e}")
                xs = np.linspace(a - 0.5, b + 0.5, 400)
                ys = [f(xi) for xi in xs]
                xs_fill = np.linspace(a, b, 300)
                ys_fill = [f(xi) for xi in xs_fill]
                fig, ax = plt.subplots(figsize=(7, 3))
                ax.plot(xs, ys, color='#4f8ef7', linewidth=2)
                ax.fill_between(xs_fill, ys_fill, alpha=0.3, color='#4f8ef7', label=f'Area={result:.4f}')
                ax.set_title("Definite Integral"); ax.legend(); ax.grid(True, alpha=0.3)
                st.pyplot(fig)
            except Exception as e:
                st.error(f"❌ Error: {e}")

    elif operation == "Gradient Descent":
        st.latex(r"\theta_{new} = \theta - \alpha \cdot f'(\theta)")
        func_input = st.text_input("f(x)", placeholder="e.g. x**2 + 3*x")
        col1, col2, col3 = st.columns(3)
        theta = col1.number_input("Starting θ", value=5.0)
        alpha = col2.number_input("Learning rate α", value=0.1, format="%.4f")
        steps = int(col3.number_input("Steps", min_value=1, max_value=500, value=20))
        if st.button("Calculate", type="primary"):
            try:
                expr, x = safe_sympy(func_input)
                deriv = sp.diff(expr, x)
                history = [theta]
                for _ in range(steps):
                    grad = float(deriv.subs(x, theta))
                    theta = theta - alpha * grad
                    history.append(theta)
                f_vals = [float(expr.subs(x, t)) for t in history]
                st.success(f"Final θ = **{theta:.6f}** | f(θ) = **{f_vals[-1]:.6f}**")
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3))
                ax1.plot(history, color='#4f8ef7', marker='o', markersize=3)
                ax1.set_title("θ over steps"); ax1.grid(True, alpha=0.3)
                ax2.plot(f_vals, color='#f74f4f', marker='o', markersize=3)
                ax2.set_title("f(θ) over steps"); ax2.grid(True, alpha=0.3)
                fig.tight_layout(); st.pyplot(fig)
            except Exception as e:
                st.error(f"❌ Error: {e}")

    elif operation == "Straight Line (y = mx + c)":
        st.latex(r"y = mx + c")
        col1, col2, col3 = st.columns(3)
        m = col1.number_input("Slope (m)", value=2.0)
        c = col2.number_input("Intercept (c)", value=1.0)
        x = col3.number_input("x value", value=3.0)
        if st.button("Calculate", type="primary"):
            y = m * x + c
            st.success(f"y = {m}×{x} + {c} = **{y}**")
            xs = np.linspace(x - 5, x + 5, 100)
            fig, ax = plt.subplots(figsize=(7, 3))
            ax.plot(xs, m*xs + c, color='#4f8ef7', linewidth=2, label=f'y={m}x+{c}')
            ax.scatter([x], [y], color='red', zorder=5, s=80, label=f'({x}, {y})')
            ax.set_title("Straight Line"); ax.legend(); ax.grid(True, alpha=0.3)
            st.pyplot(fig)

# ─────────────────────────────────────────────────────────
# 6. ML FUNCTIONS
# ─────────────────────────────────────────────────────────
elif category == "ML Functions (Loss & Activation)":

    sub = st.selectbox("Choose type", ["Loss Functions", "Activation Functions"], index=None)

    if sub == "Loss Functions":
        operation = st.selectbox("Operation",
                                 ["MSE (Mean Squared Error)",
                                  "MAE (Mean Absolute Error)",
                                  "RMSE (Root Mean Squared Error)",
                                  "Binary Cross-Entropy",
                                  "Categorical Cross-Entropy"],
                                 index=None, placeholder="choose loss function")

        st.info("💡 Enter comma-separated values. Example: `1, 2, 3`")
        y_true_input = st.text_input("y_true (actual values)", placeholder="e.g. 1, 0, 1, 1")
        y_pred_input = st.text_input("y_pred (predicted values)", placeholder="e.g. 0.9, 0.2, 0.8, 0.6")

        if st.button("Calculate", type="primary"):
            if operation and y_true_input and y_pred_input:
                try:
                    y_true = np.array([float(x.strip()) for x in y_true_input.split(',')])
                    y_pred = np.array([float(x.strip()) for x in y_pred_input.split(',')])
                    if len(y_true) != len(y_pred):
                        st.error("❌ y_true and y_pred must have same length!")
                    else:
                        if operation == "MSE (Mean Squared Error)":
                            st.latex(r"MSE = \frac{1}{n}\sum(y_{true} - y_{pred})^2")
                            result = np.mean((y_true - y_pred)**2)
                        elif operation == "MAE (Mean Absolute Error)":
                            st.latex(r"MAE = \frac{1}{n}\sum|y_{true} - y_{pred}|")
                            result = np.mean(np.abs(y_true - y_pred))
                        elif operation == "RMSE (Root Mean Squared Error)":
                            st.latex(r"RMSE = \sqrt{\frac{1}{n}\sum(y_{true} - y_{pred})^2}")
                            result = np.sqrt(np.mean((y_true - y_pred)**2))
                        elif operation == "Binary Cross-Entropy":
                            st.latex(r"BCE = -\frac{1}{n}\sum[y\log(\hat{y}) + (1-y)\log(1-\hat{y})]")
                            eps = 1e-15
                            y_pred_clip = np.clip(y_pred, eps, 1 - eps)
                            result = -np.mean(y_true * np.log(y_pred_clip) + (1 - y_true) * np.log(1 - y_pred_clip))
                        elif operation == "Categorical Cross-Entropy":
                            st.latex(r"CCE = -\sum y_{true} \log(y_{pred})")
                            eps = 1e-15
                            result = -np.sum(y_true * np.log(np.clip(y_pred, eps, 1)))
                        st.success(f"✅ {operation} = **{result:.6f}**")

                        # Error bar chart
                        errors = y_true - y_pred
                        fig, ax = plt.subplots(figsize=(7, 3))
                        ax.bar(range(len(errors)), errors,
                               color=['#4f8ef7' if e >= 0 else '#f74f4f' for e in errors])
                        ax.axhline(0, color='gray', linewidth=0.8)
                        ax.set_title("Prediction Errors (y_true - y_pred)"); ax.grid(True, alpha=0.3)
                        st.pyplot(fig)
                except Exception as e:
                    st.error(f"❌ Error: {e}")
            else:
                st.warning("⚠️ Fill in all fields!")

    elif sub == "Activation Functions":
        operation = st.selectbox("Operation",
                                 ["Sigmoid", "ReLU", "Leaky ReLU",
                                  "Tanh", "Softmax", "ELU"],
                                 index=None, placeholder="choose activation")

        x_val = st.number_input("Input value (x)", value=1.0)

        def sigmoid(x):   return 1 / (1 + np.exp(-x))
        def relu(x):      return np.maximum(0, x)
        def leaky_relu(x, alpha=0.01): return np.where(x > 0, x, alpha * x)
        def elu(x, alpha=1.0): return np.where(x > 0, x, alpha * (np.exp(x) - 1))

        if st.button("Calculate", type="primary") and operation:
            xs = np.linspace(-5, 5, 400)

            if operation == "Sigmoid":
                st.latex(r"\sigma(x) = \frac{1}{1+e^{-x}}")
                result = sigmoid(x_val)
                st.success(f"σ({x_val}) = **{result:.6f}**")
                st.info("Output range: (0, 1) — used in binary classification output layer")
                fig = plot_function(sigmoid, title="Sigmoid", label="σ(x)")

            elif operation == "ReLU":
                st.latex(r"ReLU(x) = \max(0, x)")
                result = relu(x_val)
                st.success(f"ReLU({x_val}) = **{result:.6f}**")
                st.info("Most common hidden layer activation. Fast & simple.")
                fig = plot_function(relu, title="ReLU", label="ReLU(x)")

            elif operation == "Leaky ReLU":
                st.latex(r"f(x) = \begin{cases} x & x > 0 \\ 0.01x & x \leq 0 \end{cases}")
                result = leaky_relu(x_val)
                st.success(f"LeakyReLU({x_val}) = **{result:.6f}**")
                st.info("Fixes the 'dying ReLU' problem by allowing small negative gradients.")
                fig = plot_function(leaky_relu, title="Leaky ReLU", label="LeakyReLU(x)")

            elif operation == "Tanh":
                st.latex(r"\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}")
                result = np.tanh(x_val)
                st.success(f"tanh({x_val}) = **{result:.6f}**")
                st.info("Output range: (-1, 1) — zero-centered, better than sigmoid for hidden layers.")
                fig = plot_function(np.tanh, title="Tanh", label="tanh(x)")

            elif operation == "Softmax":
                st.latex(r"\text{softmax}(x_i) = \frac{e^{x_i}}{\sum e^{x_j}}")
                st.info("💡 Enter comma-separated values for softmax (works on a vector).")
                vec_input = st.text_input("Input vector", value="1.0, 2.0, 3.0")
                vec = np.array([float(v.strip()) for v in vec_input.split(',')])
                e = np.exp(vec - np.max(vec))
                result = e / e.sum()
                st.success(f"Softmax = **{np.round(result, 6)}**")
                st.caption(f"Sum = {result.sum():.6f} (always = 1.0)")
                fig, ax = plt.subplots(figsize=(7, 3))
                ax.bar(range(len(result)), result, color='#4f8ef7', edgecolor='white')
                ax.set_title("Softmax Output Probabilities"); ax.grid(True, alpha=0.3)
                st.pyplot(fig)
                fig = None

            elif operation == "ELU":
                st.latex(r"ELU(x) = \begin{cases} x & x > 0 \\ \alpha(e^x-1) & x \leq 0 \end{cases}")
                result = elu(x_val)
                st.success(f"ELU({x_val}) = **{result:.6f}**")
                st.info("Smoother than ReLU for negative values.")
                fig = plot_function(elu, title="ELU", label="ELU(x)")

            if operation != "Softmax" and fig is not None:
                st.pyplot(fig)
                