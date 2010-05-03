def prettyprint_list(list):
	for item in list:
		print('%s' % (item, ))
	print '\n'
	
def prettyprint_map(map):
	for x in map.keys():
		print('KEY', '\t', 'VALUE')
		print(x, '\t', map[x])
		print('  key', '\t','value')
		for y in map[x].keys():
			print('  ', y, '\t', map[x][y])

def prettywrite_map(map, file):
	file.write('KEY\tVALUE\n' + str(map))
	for x in map.keys():	
		file.write('  ' + x + '\t' + str(map[x]) +'\n')
		
def prettywrite_nested_map(map, file):
	for x in map.keys():
		file.write('KEY\tVALUE\n')
		file.write(x + '\t' + str(map[x]) +'\n')
		file.write('  key\tvalue\n')
		for y in map[x].keys():
			file.write('  ' + y + '\t' + str(map[x][y]) + '\n')
