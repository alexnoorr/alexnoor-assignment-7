
import numpy as np
from flask import Flask, render_template, request, session, redirect, url_for
import matplotlib.pyplot as plt
import scipy.stats as stats
import os

app = Flask(__name__)
app.secret_key = 'your_secret_key'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    # Gather inputs
    N = int(request.form['N'])
    mu = float(request.form['mu'])
    sigma2 = float(request.form['sigma2'])
    beta0 = float(request.form['beta0'])
    beta1 = float(request.form['beta1'])
    S = int(request.form['S'])

    # Generate data and store simulations
    slopes, intercepts = [], []
    for _ in range(S):
        X = np.random.normal(mu, np.sqrt(sigma2), N)
        error = np.random.normal(0, np.sqrt(sigma2), N)
        Y = beta0 + beta1 * X + error
        # Fit linear regression
        slope, intercept = np.polyfit(X, Y, 1)
        slopes.append(slope)
        intercepts.append(intercept)

    session['slopes'] = slopes
    session['intercepts'] = intercepts
    session['beta0'] = beta0
    session['beta1'] = beta1

    # Plotting
    plt.scatter(X, Y)
    plt.plot(X, beta0 + beta1 * X, color='red')
    plt.savefig('static/plot1.png')
    plt.close()

    # Histogram of slopes and intercepts
    plt.hist(slopes, alpha=0.5, label='Slopes')
    plt.hist(intercepts, alpha=0.5, label='Intercepts')
    plt.legend()
    plt.savefig('static/plot2.png')
    plt.close()

    return redirect(url_for('index'))

@app.route('/hypothesis_test', methods=['POST'])
def hypothesis_test():
    parameter = request.form['parameter']
    test_type = request.form['test_type']
    slopes = session.get('slopes', [])
    intercepts = session.get('intercepts', [])
    hypothesized_value = session['beta1'] if parameter == 'slope' else session['beta0']
    simulated_values = slopes if parameter == 'slope' else intercepts
    observed_stat = np.mean(simulated_values)
    
    # Calculate p-value
    if test_type == '>':
        p_value = np.mean([stat >= observed_stat for stat in simulated_values])
    elif test_type == '<':
        p_value = np.mean([stat <= observed_stat for stat in simulated_values])
    else:
        p_value = np.mean([abs(stat) >= abs(observed_stat) for stat in simulated_values])

    fun_message = "Rare event detected!" if p_value <= 0.0001 else None

    # Hypothesis testing plot
    plt.hist(simulated_values, alpha=0.7)
    plt.axvline(observed_stat, color='red', linestyle='dashed', linewidth=2)
    plt.axvline(hypothesized_value, color='blue', linestyle='dotted', linewidth=2)
    plt.savefig('static/plot3.png')
    plt.close()

    return redirect(url_for('index'))

@app.route('/confidence_interval', methods=['POST'])
def confidence_interval():
    parameter = request.form['parameter']
    confidence_level = int(request.form['confidence_level'])
    simulated_values = session.get('slopes') if parameter == 'slope' else session.get('intercepts')
    mean_estimate = np.mean(simulated_values)
    se = np.std(simulated_values) / np.sqrt(len(simulated_values))
    t_value = stats.t.ppf((1 + confidence_level / 100) / 2., len(simulated_values) - 1)
    ci_lower, ci_upper = mean_estimate - t_value * se, mean_estimate + t_value * se
    includes_true = ci_lower <= session['beta1'] if parameter == 'slope' else ci_lower <= session['beta0']

    # Confidence interval plot
    plt.plot(simulated_values, 'o', color='gray')
    plt.axhline(mean_estimate, color='blue')
    plt.axhline(ci_lower, color='green', linestyle='--')
    plt.axhline(ci_upper, color='green', linestyle='--')
    plt.savefig('static/plot4.png')
    plt.close()

    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
