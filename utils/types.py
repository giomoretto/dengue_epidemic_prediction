from enum import Enum


class City(Enum):
    BELO_HORIZONTE = "Belo Horizonte"
    RIO_DE_JANEIRO = "Rio de Janeiro"
    SAO_PAULO = "São Paulo"
    VITORIA = "Vitória"


# Model feature "city_encoded": codes in alphabetical order of city names
CITY_ENCODING: dict[City, int] = {
    City.BELO_HORIZONTE: 0,
    City.RIO_DE_JANEIRO: 1,
    City.SAO_PAULO: 2,
    City.VITORIA: 3,
}
