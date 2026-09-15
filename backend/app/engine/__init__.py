"""
The Nu Score modelling engine.

train.run() takes a modelling table and returns everything the UI and
reports need: a tuned model zoo, a composite-score champion, calibrated
probabilities, Nu Score bands, SHAP explanations and a saved model bundle
that can score new applications after the training data has been deleted.
"""
