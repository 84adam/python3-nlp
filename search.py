from flask import Flask, request
import json
from random import randint
import calendar, datetime

def rand_id(n):
	n_digit_str = ''.join(["{}".format(randint(0, 9)) for num in range(0, n)])
	return int(n_digit_str)

def get_ts():
	d = datetime.datetime.utcnow()
	ts = calendar.timegm(d.timetuple())
	return ts

def gen_json(doc):
	query_text = doc
	source = "Search Bar"
	query_id = rand_id(10)
	query_ts = get_ts()
	q_dict = ({'source': f'{source}', 'query_id': f'{query_id}', 'query_ts': f'{query_ts}', 'query_text': f'{query_text}'})
	return json.dumps(q_dict)

app = Flask(__name__)

#create a 'search' route
@app.route('/search', methods=['POST', 'GET'])

# present search form, allow user submission
# return query string and JSON object with query text, source, id, and timestamp
def search():
  if request.method == 'POST':
    query = request.form.get('query')
    new_json = gen_json(query)
    return f'''<h2>You entered: "{query}"</h2><p>JSON: </p><p>{new_json}</p>'''
  return '''<form method="POST">
  Search Terms: <input type="text" name="query">
  <input type="submit">
  </form>'''

if __name__ == '__main__':
  app.run(debug=True, port=5000)
