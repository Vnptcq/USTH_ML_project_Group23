def create_features(data):
    data['savings'] = data['income'] - data['total_expense']
    data['spending_ratio'] = data['total_expense'] / data['income']
    return data
