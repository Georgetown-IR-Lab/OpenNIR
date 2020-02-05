import onir


COMPONENTS = {
    'rankers': onir.rankers.registry,
    'vocabs': onir.vocab.registry,
    'datasets': onir.datasets.registry,
    'trainers': onir.trainers.registry,
    'predictors': onir.predictors.registry,
    'pipelines': onir.pipelines.registry,
}


def main():
    for name, registry in COMPONENTS.items():
        print(name.upper())
        print('default: ' + registry.default)
        print('================')
        for cname, component in registry.registered.items():
            if cname is None:
                continue
            print(cname)
            if component.__doc__ is not None:
                print(component.__doc__)
            else:
                print('''
    (no docstring)
''')
            print('settings')
            for key, val in component.default_config().items():
                print(f'  {key}: {val}')
            print('----------------\n')
    print()


if __name__ == '__main__':
    main()
