# -------------------------------
# Common interface functions

def load_data():
    return data


def create_model():
    return model


def train_model(model, data):
    return trained_model


def test_model(model, data):
    print(results)

# -------------------------------


# -------------------------------
# Test function 1

def main_test():
    data = load_data()
    model = create_model()
    trained_model = train_model(model, data)
    test_model(trained_model, test_data)


if __name__ == "__main__":
    # assert False    # Not meant to be run as a script

    # Call the test functions here
    main_test()
