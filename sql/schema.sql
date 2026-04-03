create table if not exists documents (
  id uuid primary key,
  title text not null,
  authors text[],
  pub_year int,
  source_path text not null
);

create table if not exists passages (
  id uuid primary key,
  document_id uuid references documents(id) on delete cascade,
  page int not null,
  headings_path text[],
  text text not null
);

create table if not exists passage_entities (
  passage_id uuid references passages(id) on delete cascade,
  entity text,
  ent_type text,            -- PERSON, ORG, GPE, FAC, etc.
  norm_entity text,
  primary key (passage_id, entity)
);

create table if not exists passage_years (
  passage_id uuid references passages(id) on delete cascade,
  year int,
  primary key (passage_id, year)
);

create index if not exists idx_passages_tsv on passages using gin (to_tsvector('simple', text));
create index on passage_entities (entity, ent_type);
create index on passage_years (year);








