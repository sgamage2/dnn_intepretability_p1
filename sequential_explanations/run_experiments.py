# This file puts together a complete experiment (model building, estimating feature significance and evaluating it)
params = {}
params['model'] = 'dummy_lstm'
params['feature_signifcance'] = 'A1'
params['metric'] = 'M1'


def main():
    # Setup the set of problem, feature_sig algorithm, metric
    if params['model'] == 'dummy_lstm':
        import models.dummy_lstm_problem as problem
        import feature_significance.A1 as feature_sig
        import metrics.M1 as metric

    # Common function calls
    data = model.load_data()
    model = create_model()
    trained_model = train_model(model, data)
    test_model(trained_model, test_data)

    significance = feature_sig.estimate_feature_sig(trained_model, data)
    metric.evaluate(significance, data)


if __name__ == "__main__":
    # os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # This line disables GPU
    main()
