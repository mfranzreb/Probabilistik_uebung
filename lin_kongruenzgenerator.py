import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np
import math


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
    n_blocks = int(math.sqrt(len(norm_random_nums)))
    while True:
        if len(norm_random_nums) % n_blocks == 0:
            break
        n_blocks -= 1

    nums_per_block = len(norm_random_nums) / n_blocks
    avg_per_block = []
    total_num = 0
    for i in range(n_blocks):
        avg_per_block.append(
            np.sum(norm_random_nums[total_num : total_num + int(nums_per_block)])
            / int(nums_per_block)
        )
        total_num += int(nums_per_block)

    mu = 0.5
    sigma = math.sqrt(1 / 12)
    norm_avg_per_block = np.sort(
        np.asarray(
            [(i - mu) / (sigma / np.sqrt(nums_per_block)) for i in avg_per_block]
        )
    )

    norm_x = np.linspace(-3, 3, 100)
    norm_y = stats.norm.cdf(norm_x)
    y = np.arange(n_blocks) / (n_blocks)
    print(norm_avg_per_block)

    chi_value, sig_lvl = stats.chisquare(norm_avg_per_block, norm_x)
    significance_lvl = 0.05
    if sig_lvl < significance_lvl:
        print("Hypothesis rejected")
    else:
        print("Fail to reject hypothesis")

    print(chi_value, sig_lvl)

    # plot normal CDF
    plt.plot(norm_x, norm_y)
    plt.plot(norm_avg_per_block, y)
    plt.show()


test_stochastic_independence(
    convert_to_normalised_random_numbers(
        generate_random_number(22695477, 1, 2**32, 2**10, 10000), 2**32
    )
)
