import numpy as np
import pickle

grade = 100
m_train = 600
m_test = 100
n = 784

def end_and_print_grade(grade):
  print('=' * 79)
  if grade == 100:
    print('¡Felicidades no se detectó ningún error!')
  print(('Su nota asignada es: NOTA<<{0}>>').format(grade if grade >= 0 else 0))
  print()
  print("Recuerde que los parámetros en el diccionario all_params.dict deben de producir un modelo con una precisión mayor al 95% en el dataset de prueba, esto se revisará después de la entrega.")
  exit()

def print_error(action, error):
  print('Error al ' + action)
  print(('El error recibido fue:\n{0}\n\n').format(error))

try:
  from hw_digit_recognizer import HWDigitRecognizer as clf
except Exception as e:
  print_error('importar la clase HWDigitRecognizer -100%', e)
  end_and_print_grade(0)

try:
  train_filename = './autograder_data/mnist_train_0.01sampled.csv'
  test_filename = './autograder_data/mnist_test_0.01sampled.csv'
    # train_filename = './datasets/mnist_train.csv'
    # test_filename = './datasets/mnist_test.csv'
  clf = clf(train_filename, test_filename)
except Exception as e:
  print_error('crear una instancia de HWDigitRecognizer -100%', e)
  end_and_print_grade(0)

try:
  datasets = clf.get_datasets()
except Exception as e:
  print('obtener los datasets -100%', e)
  end_and_print_grade(0)

try:
  def check_and_save_dataset(dsName, ds, shape):
    if(ds.shape != shape):
      raise Exception("{0} no tiene las dimensiones correctas: {1} != {2}".format(
        dsName, str(ds.shape), str(shape)))
            
  check_and_save_dataset("X_train", datasets["X_train"], (n,m_train))
  check_and_save_dataset("X_test", datasets["X_test"], (n,m_test))
  check_and_save_dataset("Y_train", datasets["Y_train"], (10,m_train))
  check_and_save_dataset("Y_test", datasets["Y_test"], (10,m_test))
except Exception as e:
  print_error('revisar los datasets -100%'.format(dsName), e)
  end_and_print_grade(0)

try:
  params, costs = clf.train_model()
except Exception as e:
  print_error('entrenar el modelo -80%', e)
  end_and_print_grade(20)

try:
  with open("autograder_data/test_params.dict", "rb") as f:
    predictions = clf.predict(clf.get_datasets()["X_test"], pickle.load(f))
    # DEBUG
    # np.savetxt("predictions.txt", predictions)
    # np.savetxt("test_predictions.txt", np.load("autograder_data/test_predictions.npy"))
    if not np.allclose(np.load("autograder_data/test_predictions.npy"), predictions):
      raise Exception('Error en los resultados retornados por la función predict')
except Exception as e:
  print_error('revisar método predict -40%', e)
  grade -= 40

try:
  predictions = clf.predict(clf.get_datasets()["X_test"], params)
  accuracy = (predictions == np.argmax(clf.get_datasets()["Y_test"], axis=0)).mean() * 100
  if accuracy < 60.0:
    raise Exception('La precisión que retorna el modelo es muy baja')
except Exception as e:
  print_error('revisar resultados del modelo entrenado -40%', e)
  grade -= 40

try:
  def check_key(key, dict):
    if not key in dict:
      raise Exception(f'No se encontró la clave {key} en el diccionario')

  with open("all_params.dict", "rb") as f:
    all_params = pickle.load(f)
    check_key("model_params", all_params)
    check_key("layer_dims", all_params)
    if all_params["layer_dims"][0] != 784:
      raise Exception('La dimensión en <layer_dims> de la primera capa es incorrecta')
    if all_params["layer_dims"][-1] != 10:
      raise Exception('La dimensión en <layer_dims> de la última capa es incorrecta')
    check_key("learning_rate", all_params)
    check_key("num_iterations", all_params)
    check_key("costs", all_params)
except Exception as e:
  print_error('revisar el archivo all_params.dict -30%', e)
  grade -= 30

end_and_print_grade(grade)