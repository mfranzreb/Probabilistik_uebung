import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np


def generate_random_number(a, b, m, x0, count):
    random_numbers = []
    for i in range(count):
        x0 = (a * x0 + b) % m
        random_numbers.append(x0)
    return random_numbers


def convert_to_normalised_random_numbers(
    random_numbers, max
):  # max should be equal to m
    for index, number in enumerate(random_numbers):
        random_numbers[index] = number / max

    return random_numbers


def test_even_spread(normalised_random_numbers, num_categories):
    n, bins, _ = plt.hist(normalised_random_numbers, bins=num_categories)
    plt.show()
    return n


def chi_squared_test(normalised_random_numbers, significance_lvl):
    result = test_even_spread(normalised_random_numbers, 100)
    expected_result = [np.average(result) for i in range(len(result))]
    chi_value, sig_lvl = stats.chisquare(result, expected_result)
    if sig_lvl < significance_lvl:
        print("Hypothesis rejected")
    else:
        print("Fail to reject hypothesis")

    print(chi_value, sig_lvl)


# chi_squared_test(
#    convert_to_normalised_random_numbers(
#        generate_random_number(22695477, 1, 2**32, 2**10, 1000000), 2**32
#    ),
#    0.05,
# )


def test_stochastic_independence(norm_random_nums):
    nums_per_bin = test_even_spread(norm_random_nums, 10)
    norm_random_nums = np.sort(np.asarray(norm_random_nums))
    avg_per_bin = []
    total_num = 0
    for num in nums_per_bin:
        avg_per_bin.append(
            np.sum(norm_random_nums[total_num : total_num + int(num)]) / int(num)
        )
        total_num += int(num)

    norm_x = np.linspace(0, 1, 1000)
    norm_y = stats.norm.cdf(norm_x)
    y = np.arange(len(norm_random_nums)) / (len(norm_random_nums) - 1)

    # plot normal CDF
    plt.plot(norm_x, norm_y)
    plt.plot(norm_random_nums, y)
    plt.show()


# test_stochastic_independence(
#    convert_to_normalised_random_numbers(
#        generate_random_number(22695477, 1, 2**32, 2**10, 1000000), 2**32
#    )
# )
