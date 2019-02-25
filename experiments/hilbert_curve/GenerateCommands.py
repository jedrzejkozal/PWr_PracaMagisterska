

class GenerateCommands(object):

        def generate_commands(self, n):
            return self.generate_commands_A(n)


        def append_list_at_index(self, source, to_add, index):
            tmp = source[index+1:]
            source = source[:index]
            source.extend(to_add)
            source.extend(tmp)

            return source


        def generate_commands_A(self, n):
            if n == 1:
                return ["left", "F", "right", "F", "right", "F", "left"]

            commands = ["left", "B", "F", "right", "A", "F", "A", "right", "F", "B", "left"]
            return self.determine_commands(commands, n)


        def generate_commands_B(self, n):
            if n == 1:
                return ["right", "F", "left", "F", "left", "F", "right"]

            commands = ["right", "A", "F", "left", "B", "F", "B", "left", "F", "A", "right"]
            return self.determine_commands(commands, n)


        def determine_commands(self, commands, n):
            i = 0
            while commands.count("A") > 0 or commands.count("B") > 0:
                if commands[i] is "A":
                    commands = self.append_list_at_index(commands, self.generate_commands_A(n-1), i)
                elif commands[i] is "B":
                    commands = self.append_list_at_index(commands, self.generate_commands_B(n-1), i)
                i += 1

            return commands
