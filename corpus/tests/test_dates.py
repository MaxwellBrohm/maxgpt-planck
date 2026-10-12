"""dates.py: every date shape seen in the core sources, the partial-date gate and the URL rule."""
import pytest

import dates as D


@pytest.mark.parametrize("value,order,want", [
    ("2016-10-09T23:39:12Z", None, "2016-10-09"),          # youtube published_time
    ("2024-02-05 08:57:07", None, "2024-02-05"),
    ("09-30-2024", "mdy", "2024-09-30"),                    # pressbooks
    ("04-9-2023", "mdy", "2023-04-09"),
    ("08/27/2020", "mdy", "2020-08-27"),                    # oercommons
    ("08/27/2020", None, None),                             # numeric needs a declared order
    ("13/01/2020", "mdy", None),                            # month 13
    ("Mar 26, 2024", None, "2024-03-26"),                   # pdr
    ("Published on December 13, 2021", None, "2021-12-13"),
    ("1st December 2022", None, "2022-12-01"),
    ("10th April 2017 / 6:18 pm / La", None, "2017-04-10"),
    ("Oct. 20, 2023", None, "2023-10-20"),
    ("Sept. 3, 2019", None, "2019-09-03"),
    ("26 July 2015 at 18:13", None, "2015-07-26"),
    ("Sunday, 26 May 2024", None, "2024-05-26"),
    ("Published: August 18, 2017 4:05 pm", None, "2017-08-18"),
    ("February 30, 2020", None, None),                      # not a calendar day
    ("08 Jul", None, None),                                 # no year
    ("4. juna, 2014.", None, None),                         # Bosnian month
    ("Activity/Lab", None, None),
    (1862, None, "1862"), ("1921", None, "1921"), (0, None, None), (True, None, None),
    ("2022-11", None, "2022-11"), ("", None, None), (None, None, None),
])
def test_parse_date(value, order, want):
    assert D.parse_date(value, order) == want


@pytest.mark.parametrize("partial,want", [
    ("2022-11-30", "ok"), ("2022-12-01", "post_cutoff"), ("2022-11", "ok"),
    ("2022-12", "post_cutoff"), ("2021", "ok"), ("2022", "post_cutoff"), ("1862", "ok"),
])
def test_gate_uses_the_latest_day_a_partial_date_can_mean(partial, want):
    assert D.gate(partial) == want


def test_url_date_and_host_rule():
    assert D.url_date("https://globalvoices.org/2004/10/26/about-x/") == "2004-10-26"
    assert D.url_date("https://oxpeckers.org/2013/07/rhino/") == "2013-07"
    assert D.url_date("https://example.org/2019/13/01/x/") is None
    assert D.url_date("https://360info.org/2021-was-a-year/") is None
    link = "http://247wallst.com/special-report/2017/08/18/black-and-white/"
    assert D.url_date(link) == "2017-08-18"
    assert D.url_date(link, host_must_contain="milwaukeenns") is None
    assert D.url_date("https://www.globalvoices.org/2004/10/26/x/", "globalvoices") == "2004-10-26"


def test_classify_order_and_fallbacks():
    gv = "https://globalvoices.org/2022/12/02/x/"
    assert D.classify("", url=gv, url_host_hint="globalvoices") == ("post_cutoff", "2022-12-02", "url")
    assert D.classify("", url="https://globalvoices.org/about/") == ("undated", None, None)
    assert D.classify("Jan 3, 2020", url=gv) == ("ok", "2020-01-03", "field")   # field wins
    assert D.classify("08 Jul", url="https://oxpeckers.org/2013/07/r/",
                      url_host_hint="oxpeckers") == ("ok", "2013-07", "url")
    assert D.classify("08 Jul") == ("unparseable", None, None)
    assert D.classify("Activity/Lab", numeric_order="mdy", nondate_is_undated=True) == (
        "undated", None, None)
    assert D.classify("Activity/Lab", numeric_order="mdy") == ("unparseable", None, None)
    assert D.classify("13/45/2020", numeric_order="mdy", nondate_is_undated=True)[0] == "unparseable"
    assert D.classify(2023) == ("post_cutoff", "2023", "field")
    assert D.classify(None) == ("undated", None, None)
