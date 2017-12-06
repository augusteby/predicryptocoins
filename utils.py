def prime_factors(n):
    i = 2
    factors = []
    while i * i <= n:
        if n % i:
            i += 1
        else:
            n //= i
            factors.append(i)
    if n > 1:
        factors.append(n)
    return list(set(factors))


def user_select_train_batch_size(possible_values):
    mess = ('Please select a value of batch size'
            ' to fit the model among the following values: %s') % str(possible_values)
    answer = input(mess)

    assert answer in possible_values, 'input value should be in the list'
    return answer

if __name__ == '__main__':
    print(user_select_train_batch_size([1,2,3]))
