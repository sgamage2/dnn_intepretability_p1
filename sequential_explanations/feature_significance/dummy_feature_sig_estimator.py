# -------------------------------
# Common interface functions

def estimate_feature_sig(model, data):
    return data

# -------------------------------


# -------------------------------
# Test function 1

def main_test():
    data = load_data()
    model = get_some_trained_model()
    featur_sig = estimate_feature_sig(model, data)
    plot_feature_sig(featur_sig)    # Should come from a utility function


if __name__ == "__main__":
    # assert False    # Not meant to be run as a script

    # Call the test functions here
    main_test()
