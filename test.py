import question_1
import question_2
from cse163_utils import assert_equals

FILE_NAME = "GLOB.SES.csv"
FILE_NAME_2 = 'global_unemployment_rate.csv'


def test_top_and_bottom_countries():
    assert_equals(['Qatar', 'United Arab Emirates', 'Kuwait',
                   'Luxembourg', 'Brunei Darussalam', 'Switzerland',
                   'Norway', 'United States', 'Denmark', 'Singapore',
                   'Uganda', 'Nepal', 'Central African Republic',
                   'Burkina Faso', 'Rwanda', 'Afghanistan', 'Niger',
                   'Congo, Dem Rep', 'Malawi', 'Ethiopia'],
                  question_1.sub_q1(FILE_NAME))


def test_yrseduc_of_adult_countries():
    assert_equals(['Argentina', 'Australia', 'Austria',
                   'Belgium', 'Bulgaria', 'Canada', 'Chile',
                   'Costa Rica', 'Cuba', 'Denmark', 'Finland',
                   'France', 'Germany', 'Greece', 'Guyana', 'Hungary',
                   'Ireland', 'Italy', 'Jamaica', 'Japan', 'Malaysia',
                   'Mexico', 'Netherlands', 'New Zealand', 'Norway',
                   'Panama', 'Paraguay', 'Peru', 'Philippines',
                   'Portugal', 'South Africa', 'Spain', 'Sweden',
                   'Switzerland', 'United Kingdom', 'United States',
                   'Uruguay', 'Venezuela'],
                  question_1.sub_q2(FILE_NAME))


def test_ideal_unemployment_rate_countries(ses_data):
    assert_equals(['Angola', 'Austria', 'Bahrain', 'Bangladesh',
                   'Benin', 'Bolivia', 'Burkina Faso', 'Burundi',
                   'Cambodia', 'China', 'Cuba', 'Cyprus', 'Ecuador',
                   'Ethiopia', 'Fiji', 'Guatemala', 'Honduras',
                   'Iceland', 'Japan', 'Kenya', 'Kuwait', 'Liberia',
                   'Luxembourg', 'Madagascar', 'Malaysia', 'Maldives',
                   'Mexico', 'Mozambique', 'Myanmar', 'Nepal',
                   'Netherlands', 'Niger', 'Nigeria', 'Norway',
                   'Pakistan', 'Panama', 'Papua New Guinea', 'Peru',
                   'Philippines', 'Qatar', 'Rwanda', 'Sierra Leone',
                   'Singapore', 'Switzerland', 'Tanzania', 'Thailand',
                   'Togo', 'Tonga', 'Uganda', 'United Arab Emirates',
                   'Vietnam'],
                  question_1.sub_q3(FILE_NAME_2, ses_data))


def main():
    test_top_and_bottom_countries()
    test_yrseduc_of_adult_countries()
    test_ideal_unemployment_rate_countries(question_1.sub_q2.ses_by_coun)


if __name__ == '__main__':
    main()
