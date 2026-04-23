from f1_prediction_ml.pipelines.build_inference_features import build_inference_features
from f1_prediction_ml.modeling.predict_winner import WinnerPredictor

def predict_cli(year: int, grand_prix: str):
    print(type(year), type(grand_prix))
    next_race_df = build_inference_features(
        year=year,
        grand_prix=grand_prix,
    )

    predictor = WinnerPredictor()

    predictions = predictor.predict_next_race_winner(next_race_df)

    print('\nPredicted probabilities:')
    print(predictions[['abbreviation', 'win_proba']])

    print('\nTop 3 predicted winners:')
    print(predictions[['event_id', 'abbreviation', 'win_proba']].head(3))

if __name__ == '__main__':
    year = input('Enter the year of the race: ')
    grand_prix = input('Enter the name of the grand prix: ')
    predict_cli(int(year), str(grand_prix))
