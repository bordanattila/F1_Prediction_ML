from f1_prediction_ml.pipelines.build_inference_features import build_inference_features
from f1_prediction_ml.modeling.predict_winner import WinnerPredictor

next_race_df = build_inference_features(
    year=2023,
    grand_prix='Hungary',
    session_types=['FP1', 'FP2', 'FP3', 'Q'],
)

predictor = WinnerPredictor()

predictions = predictor.predict_next_race_winner(next_race_df)
winner = predictor.get_predicted_winner(predictions)

print('\nPredicted probabilities:')
print(predictions[['abbreviation', 'win_proba']])

print('\nPredicted winner:')
print(winner[['event_id', 'abbreviation', 'win_proba']])
