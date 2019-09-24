
def my_stop_words():
    stop_words = ['a','ao', 'aos', 'aquela', 'aquelas', 'aquele', 'aqueles', 'aquilo',\
           'as', 'até', 'com', 'como', 'da', 'das', 'de', 'dela', 'delas',\
           'dele', 'deles', 'depois', 'do', 'dos', 'e', 'ela', 'elas', 'ele',\
           'eles', 'em', 'entre', 'era', 'eram', 'essa', 'essas', 'esse',\
           'esses', 'esta', 'estamos', 'estas', 'estava', 'estavam', 'este',\
           'esteja', 'estejam', 'estejamos', 'estes', 'esteve', 'estive',\
           'estivemos', 'estiver', 'estivera', 'estiveram', 'estiverem',\
           'estivermos', 'estivesse', 'estivessem', 'estivéramos',\
           'estivéssemos', 'estou', 'está', 'estávamos', 'estão', 'eu', 'foi',\
           'fomos', 'for', 'fora', 'foram', 'forem', 'formos', 'fosse',\
           'fossem', 'fui', 'fôramos', 'fôssemos', 'haja', 'hajam', 'hajamos',\
           'havemos', 'hei', 'houve', 'houvemos', 'houver', 'houvera',\
           'houveram', 'houverei', 'houverem', 'houveremos', 'houveria',\
           'houveriam', 'houvermos', 'houverá', 'houverão', 'houveríamos',\
           'houvesse', 'houvessem', 'houvéramos', 'houvéssemos', 'há', 'hão',\
           'isso', 'isto', 'já', 'lhe', 'lhes', 'mais', 'mas', 'me', 'mesmo',\
           'meu', 'meus', 'minha', 'minhas', 'na', 'nas', 'nem',\
           'no', 'nos', 'nossa', 'nossas', 'nosso', 'nossos', 'num', 'numa',\
           'nós', 'o', 'os', 'ou', 'para', 'pela', 'pelas', 'pelo',\
           'pelos', 'por', 'qual', 'quando', 'que', 'quem', 'se', 'seja',\
           'sejam', 'sejamos', 'sem', 'serei', 'seremos', 'seria', 'seriam',\
           'será', 'serão', 'seríamos', 'seu', 'seus', 'somos', 'sou', 'sua',\
           'suas', 'são', 'só', 'também', 'te', 'tem', 'temos', 'tenha',\
           'tenham', 'tenhamos', 'tenho', 'terei', 'teremos', 'teria',\
           'teriam', 'terá', 'terão', 'teríamos', 'teu', 'teus', 'teve',\
           'tinha', 'tinham', 'tive', 'tivemos', 'tiver', 'tivera', 'tiveram',\
           'tiverem', 'tivermos', 'tivesse', 'tivessem', 'tivéramos',\
           'tivéssemos', 'tu', 'tua', 'tuas', 'tém', 'tínhamos', 'um', 'uma',\
           'você', 'vocês', 'vos', 'à', 'às', 'éramos', 'agora', 'ainda',\
           'alguém', 'algum', 'alguma', 'algumas', 'alguns', 'ampla',\
           'amplas', 'amplo', 'amplos', 'ante', 'antes', 'ao', 'aos', 'após',\
           'aquela', 'aquelas', 'aquele', 'aqueles', 'aquilo', 'as', 'até',\
           'através', 'cada', 'coisa', 'coisas', 'com', 'como', 'contra',\
           'contudo', 'da', 'daquele', 'daqueles', 'das', 'de', 'dela',\
           'delas', 'dele', 'deles', 'depois', 'dessa', 'dessas', 'desse',\
           'desses', 'desta', 'destas', 'deste', 'deste', 'destes', 'deve',\
           'devem', 'devendo', 'dever', 'deverá', 'deverão', 'deveria',\
           'deveriam', 'devia', 'deviam', 'disse', 'disso', 'disto', 'dito',\
           'diz', 'dizem', 'do', 'dos', 'e', 'ela', 'elas', 'ele',\
           'eles', 'em', 'enquanto', 'entre', 'era', 'essa', 'essas', 'esse',\
           'esses', 'esta', 'está', 'estamos', 'estão', 'estas', 'estava',\
           'estavam', 'estávamos', 'este', 'estes', 'estou', 'eu', 'fazendo',\
           'fazer', 'feita', 'feitas', 'feito', 'feitos', 'foi', 'for',\
           'foram', 'fosse', 'fossem', 'grande', 'grandes', 'há', 'isso',\
           'isto', 'já', 'la', 'lá', 'lhe', 'lhes', 'lo', 'mas', 'me',\
           'mesma', 'mesmas', 'mesmo', 'mesmos', 'meu', 'meus', 'minha',\
           'minhas', 'na', 'nas',\
           'nem', 'nenhum', 'nessa', 'nessas', 'nesta', 'nestas', 'ninguém',\
           'no', 'nos', 'nós', 'nossa', 'nossas', 'nosso', 'nossos', 'num',\
           'numa', 'nunca', 'o', 'os', 'ou', 'outra', 'outras', 'outro',\
           'outros', 'para', 'pela', 'pelas', 'pelo', 'pelos', 'pequena',\
           'pequenas', 'pequeno', 'pequenos', 'perante', 'pode',\
           'pude', 'podendo', 'poder', 'poderia', 'poderiam', 'podia',\
           'podiam', 'pois', 'por', 'porém', 'porque', 'posso', 'pouca',\
           'poucas', 'pouco', 'poucos', 'primeiro', 'primeiros', 'própria',\
           'próprias', 'próprio', 'próprios', 'quais', 'qual', 'quando',\
           'quanto', 'quantos', 'que', 'quem', 'são', 'se', 'seja', 'sejam',\
           'sem', 'sempre', 'sendo', 'será', 'serão', 'seu', 'seus', 'si',\
           'sido', 'só', 'sob', 'sobre', 'sua', 'suas', 'talvez', 'também',\
           'tampouco', 'te', 'tem', 'tendo', 'tenha', 'ter', 'teu', 'teus',\
           'ti', 'tido', 'tinha', 'tinham', 'toda', 'todas', 'todavia',\
           'todo', 'todos', 'tu', 'tua', 'tuas', 'tudo', 'última', 'últimas',\
           'último', 'últimos', 'um', 'uma', 'umas', 'uns', 'vendo', 'ver',\
           'vez', 'vindo', 'vir', 'vos', 'vós']
    return stop_words