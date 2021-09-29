


class InitializationException(Exception):
    """当一个函数在系统初始化之前运行时，则会抛出该异常"""

    def __init__(self,
                 message='当前系统配置为none. 你执行过系统初始化吗？'):
        super().__init__(message)


class DoesNotExistException(Exception):
    """当系统中不存在 `name`，但正在执行需要它存在的操作时，引发异常。"""

    def __init__(self,
                 name='',
                 reason='',
                 message='{} does not exist! This might be due to: {}'):
        super().__init__(message.format(name, reason))


class AlreadyExistsException(Exception):
    """当系统中已经存在 `name` ，但操作正在尝试创建具有相同名称的资源时，引发异常。"""

    def __init__(self,
                 name='',
                 resource_type=''):
        message = f'{resource_type} `{name}` already exists! Please use ' \
                  f'Repository.get_instance().get_{resource_type}_by_name' \
                  f'("{name}") to fetch it.'
        super().__init__(message)


class PipelineNotSucceededException(Exception):
    """尝试从未成功的流水线拉取制品时，引发异常。"""

    def __init__(self,
                 name='',
                 message='{} is not yet completed successfully.'):
        super().__init__(message.format(name))



