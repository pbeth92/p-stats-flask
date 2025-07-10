class SchemaUtils:
    @staticmethod
    def to_camel_case(snake_str):
        components = snake_str.split('_')
        return components[0] + ''.join(x.title() for x in components[1:])
    
    @staticmethod
    def recursive_camelized_element(value, **kwargs):
        if isinstance(value, dict):
            return {SchemaUtils.to_camel_case(k): SchemaUtils.recursive_camelized_element(v, **kwargs) for k, v in value.items()}

        elif isinstance(value, list):
            return [SchemaUtils.recursive_camelized_element(v, **kwargs) for v in value]

        else:
            return value