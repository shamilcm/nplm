import numpy
import numpy.random
import scipy.sparse

def diag_dot(a, b, out=None):
    """Input:  a and b are arrays.
       Output: a column vector of the dot product of the rows of a and respective 
       columns of b, in other words, diag(a.dot(b))."""
    if out is None:
        out = numpy.empty((a.shape[0], 1))
    numpy.einsum('ji,ij->j', a, b, out=out[:,0])
    return out

class NeuralLM(object):
    def __init__(self, ngram_size, input_vocab_size, output_vocab_size, input_embedding_dimension, num_hidden, output_embedding_dimension, activation_function):
        self.input_vocab_size = input_vocab_size
        self.output_vocab_size = output_vocab_size
        self.index_to_word = []
        self.index_to_word_input = []
        self.index_to_word_output = []
        self.word_to_index = {}
        self.word_to_index_input = {}
        self.word_to_index_output = {}

        self.ngram_size = ngram_size
        self.input_embedding_dimension = input_embedding_dimension
        self.num_hidden = num_hidden
        self.output_embedding_dimension = output_embedding_dimension
        self.activation_function = activation_function

        self.input_embeddings = numpy.zeros((input_vocab_size,      input_embedding_dimension))
        if not num_hidden:
            self.hidden1_weights  = numpy.zeros((output_embedding_dimension,            (ngram_size-1)*input_embedding_dimension))
            self.hidden2_weights  = numpy.zeros((1, 1))
        else:
            self.hidden1_weights  = numpy.zeros((num_hidden,            (ngram_size-1)*input_embedding_dimension))
            self.hidden2_weights  = numpy.zeros((output_embedding_dimension, num_hidden))
        self.output_weights   = numpy.zeros((output_vocab_size,     output_embedding_dimension))
        self.output_biases    = numpy.zeros((output_vocab_size,     1))

    def initialize(self, r):
        def uniform(m):
            m[:,:] = numpy.random.uniform(-r, r, m.shape)
        uniform(self.input_embeddings)
        uniform(self.hidden1_weights)
        uniform(self.hidden2_weights)
        uniform(self.output_weights)
        uniform(self.output_biases)

    def forward_prop(self, inputs, output=None, normalize=True):
        u = numpy.bmat([[self.input_embeddings.T * ui] for ui in inputs])
        h1 = numpy.maximum(0., self.hidden1_weights * u)
        if not self.num_hidden:
            h2 = h1
        else:
            h2 = numpy.maximum(0., self.hidden2_weights * h1)

        if output is None:
            o = self.output_weights * h2 + self.output_biases
        else:
            # Inefficient version:
            #o = diag_dot(output.T, (self.output_weights * h2 + self.output_biases))
            #o = output.multiply(self.output_weights * h2 + self.output_biases)

            # Since output is sparse, distributing multiplication by output
            # is much more efficient:

            o = diag_dot(output.T * self.output_weights, h2) + output.T * self.output_biases
        return o

    def backward_prop(self, g_output):
        pass

    def to_file(self, outfile):

        def write_matrix(m):
            for i in xrange(m.shape[0]):
                outfile.write("\t".join(map(str, m[i])))
                outfile.write("\n")
            outfile.write("\n")

        def write_vector(m):
            for i in xrange(m.shape[0]):
                outfile.write(str(m[i]))
                outfile.write("\n")
            outfile.write("\n")

        outfile.write("\\config\n")
        outfile.write("version 1\n")
        outfile.write("ngram_size %d\n" % self.ngram_size)
        outfile.write("input_vocab_size %d\n" % self.input_vocab_size)
        outfile.write("output_vocab_size %d\n" % self.output_vocab_size)
        outfile.write("input_embedding_dimension %d\n" % self.input_embedding_dimension)
        outfile.write("num_hidden %d\n" % self.num_hidden)
        outfile.write("output_embedding_dimension %d\n" % self.output_embedding_dimension)
        outfile.write("activation_function %s\n" % self.activation_function)
        outfile.write("\n")

        if self.index_to_word_input and self.index_to_word_output:
            outfile.write("\\input_vocab\n")
            for word in self.index_to_word_input:
                outfile.write(word + "\n")
            outfile.write("\n")

            outfile.write("\\output_vocab\n")
            for word in self.index_to_word_output:
                outfile.write(word + "\n")
            outfile.write("\n")

        elif self.index_to_word:
            outfile.write("\\vocab\n")
            for word in self.index_to_word:
                outfile.write(word + "\n")
            outfile.write("\n")

        outfile.write("\\input_embeddings\n")
        write_matrix(self.input_embeddings)

        outfile.write("\\hidden_weights 1\n")
        write_matrix(self.hidden1_weights)

        outfile.write("\\hidden_weights 2\n")
        write_matrix(self.hidden2_weights)

        outfile.write("\\output_weights\n")
        write_matrix(self.output_weights)

        outfile.write("\\output_biases\n")
        write_matrix(self.output_biases)

        outfile.write("\\end\n")

    @staticmethod
    def from_file(infile):
        """Create a NeuralLM from a text file."""

        # Helper functions
        def read_sections(infile):
            while True:
                line = infile.next().strip()
                if line == "\\end":
                    break
                elif line.startswith('\\'):
                    yield line, read_section(infile)

        def read_section(infile):
            while True:
                line = infile.next().strip()
                if line == "":
                    break
                else:
                    yield line

        def read_matrix(lines, m, n, out=None):
            if out is None:
                out = numpy.zeros((m, n))
            i = 0
            for line in lines:
                row = numpy.array(map(float, line.split()))
                if len(row) != n:
                    raise Exception("wrong number of columns (expected %d, found %d)" % (n, len(row)))
                if i >= m:
                    raise Exception("wrong number of rows (expected %d, found more)" % m)
                out[i,:] = row
                i += 1
            if i < m:
                raise Exception("wrong number of rows (expected %d, found %d)" % (m, i))
            return out

        if isinstance(infile, str):
            infile = open(infile)

        for section, lines in read_sections(infile):
            if section == "\\config":
                config = {}
                for line in lines:
                    key, value = line.split()
                    config[key] = value

                m = NeuralLM(ngram_size=int(config['ngram_size']),
                             input_vocab_size=int(config['input_vocab_size']),
                             output_vocab_size=int(config['output_vocab_size']),
                             input_embedding_dimension=int(config['input_embedding_dimension']),
                             num_hidden=int(config['num_hidden']),
                             output_embedding_dimension=int(config['output_embedding_dimension']),
                             activation_function=config['activation_function'])

            elif section == "\\input_vocab":
                for line in lines:
                    m.index_to_word_input.append(line)
                m.word_to_index_input = dict((w,i) for (i,w) in enumerate(m.index_to_word_input))

            elif section == "\\output_vocab":
                for line in lines:
                    m.index_to_word_output.append(line)
                m.word_to_index_output = dict((w,i) for (i,w) in enumerate(m.index_to_word_output))

            elif section == "\\vocab":
                for line in lines:
                    m.index_to_word.append(line)
                m.word_to_index = dict((w,i) for (i,w) in enumerate(m.index_to_word))

            elif section == "\\input_embeddings":
                read_matrix(lines, m.input_vocab_size, m.input_embedding_dimension, out=m.input_embeddings)
            elif section == "\\hidden_weights 1":
                if not m.num_hidden:
                    read_matrix(lines, m.output_embedding_dimension, (m.ngram_size-1)*m.input_embedding_dimension, out=m.hidden1_weights)
                else:
                    read_matrix(lines, m.num_hidden, (m.ngram_size-1)*m.input_embedding_dimension, out=m.hidden1_weights)
            elif section == "\\hidden_weights 2":
                if not m.num_hidden:
                    read_matrix(lines, 1, 1, out=m.hidden2_weights)
                else:
                    read_matrix(lines, m.output_embedding_dimension, m.num_hidden, out=m.hidden2_weights)
            elif section == "\\output_weights":
                read_matrix(lines, m.output_vocab_size, m.output_embedding_dimension, out=m.output_weights)
            elif section == "\\output_biases":
                read_matrix(lines, m.output_vocab_size, 1, out=m.output_biases)
        return m

    def make_data(self, ngrams):
        """Takes a list of n-grams of words (as ints),
           and converts into a list of n sparse arrays."""
        rows = [[] for j in xrange(self.ngram_size)]
        cols = [[] for j in xrange(self.ngram_size)]
        values = [[] for j in xrange(self.ngram_size)]
        for i, ngram in enumerate(ngrams):
            for j, w in enumerate(ngram):
                rows[j].append(w)
                cols[j].append(i)
                values[j].append(1)
        data = [scipy.sparse.csc_matrix((values[j], (rows[j], cols[j])), shape=(self.input_vocab_size, len(ngrams))) for j in xrange(self.ngram_size-1)]
        data.append(scipy.sparse.csc_matrix((values[-1], (rows[-1], cols[-1])), shape=(self.output_vocab_size, len(ngrams))))
        return data
