from generate import Generator

data = "Console.WriteLine(\"Hello World\");"
gen = Generator()
result = gen.generate(data)
print(result)