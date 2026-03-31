---
layout: default
permalink: /blog/
title: blog
nav: true
nav_order: 2
pagination:
  enabled: true
  collection: posts
  permalink: /page/:num/
  per_page: 5
  sort_field: importance
  sort_reverse: false
  trail:
    before: 3 # The number of links before the current page
    after: 3 # The number of links after the current page -->
---

<div class="post">
  <style>
    .post-list-thumb {
      width: 100%;
      height: auto;
      max-height: 240px;
      object-fit: contain;
      object-position: center;
      display: block;
    }

    .post-thumb-col {
      display: flex;
      align-items: center;
      justify-content: center;
    }

    .post-links a {
      margin-right: 0.6rem;
      text-decoration: none;
    }
  </style>

{% assign blog_name_size = site.blog_name | size %}
{% assign blog_description_size = site.blog_description | size %}

{% if blog_name_size > 0 or blog_description_size > 0 %}
  <div class="header-bar">
    <h1>{{ site.blog_name }}</h1>
    <h2>{{ site.blog_description }}</h2>
  </div>
{% endif %}

{% if site.display_tags or site.display_categories %}
  <div class="tag-category-list">
    <ul class="p-0 m-0">
      {% for tag in site.display_tags %}
        <li>
          <i class="fa-solid fa-hashtag fa-sm"></i> <a href="{{ tag | slugify | prepend: '/blog/tag/' | relative_url }}">{{ tag }}</a>
        </li>
        {% unless forloop.last %}
          <p>&bull;</p>
        {% endunless %}
      {% endfor %}
      {% if site.display_categories.size > 0 and site.display_tags.size > 0 %}
        <p>&bull;</p>
      {% endif %}
      {% for category in site.display_categories %}
        <li>
          <i class="fa-solid fa-tag fa-sm"></i> <a href="{{ category | slugify | prepend: '/blog/category/' | relative_url }}">{{ category }}</a>
        </li>
        {% unless forloop.last %}
          <p>&bull;</p>
        {% endunless %}
      {% endfor %}
    </ul>
  </div>
{% endif %}

{% assign featured_posts = site.posts | where: "featured", "true" %}
{% if featured_posts.size > 0 %}
<br>

<div class="container featured-posts">
  {% assign is_even = featured_posts.size | modulo: 2 %}
  <div class="row row-cols-{% if featured_posts.size <= 2 or is_even == 0 %}2{% else %}3{% endif %}">
    {% for post in featured_posts %}
      <div class="col mb-4">
        <div class="card hoverable">
          <div class="row g-0">
            <div class="col-md-12">
              <div class="card-body">
                <div class="float-right">
                  <i class="fa-solid fa-thumbtack fa-xs"></i>
                </div>

                <h3 class="card-title text-lowercase">
                  {% if post.redirect == blank %}
                    <a class="post-title" href="{{ post.url | relative_url }}">{{ post.title }}</a>
                  {% elsif post.redirect contains '://' %}
                    <a class="post-title" href="{{ post.redirect }}" target="_blank" rel="noopener noreferrer">{{ post.title }}</a>
                  {% else %}
                    <a class="post-title" href="{{ post.redirect | relative_url }}">{{ post.title }}</a>
                  {% endif %}
                </h3>

                <p class="card-text">{{ post.description }}</p>

                {% if post.community_article_url or post.blogpost_url or post.thesis_url or post.paper_url or post.code_url %}
                  <p class="post-links">
                    {% if post.community_article_url %}
                      <a href="{{ post.community_article_url }}" title="Community Article" target="_blank" rel="noopener noreferrer">
                        <i class="fa-brands fa-hugging-face"></i>
                      </a>
                    {% endif %}
                    {% if post.blogpost_url %}
                      <a href="{{ post.blogpost_url }}" title="Blogpost" target="_blank" rel="noopener noreferrer">
                        <i class="fa-regular fa-newspaper"></i>
                      </a>
                    {% endif %}
                    {% if post.thesis_url %}
                      <a href="{{ post.thesis_url }}" title="Thesis" target="_blank" rel="noopener noreferrer">
                        <i class="fa-solid fa-graduation-cap"></i>
                      </a>
                    {% endif %}
                    {% if post.paper_url %}
                      <a href="{{ post.paper_url }}" title="Paper" target="_blank" rel="noopener noreferrer">
                        <i class="fa-solid fa-file-lines"></i>
                      </a>
                    {% endif %}
                    {% if post.code_url %}
                      <a href="{{ post.code_url }}" title="Code" target="_blank" rel="noopener noreferrer">
                        <i class="fa-brands fa-github"></i>
                      </a>
                    {% endif %}
                  </p>
                {% endif %}

                {% if post.external_source == blank %}
                  {% assign text_body = post.content | split: '<div id="references-section">' | first %}
                  {% assign words = text_body | number_of_words %}
                  {% assign read_time = words | divided_by: 180 | plus: 1 %}
                {% else %}
                  {% assign read_time = post.feed_content | strip_html | number_of_words | divided_by: 180 | plus: 1 %}
                {% endif %}
                {% assign year = post.date | date: "%Y" %}

                <p class="post-meta">
                  {{ read_time }} min read &nbsp; &middot; &nbsp;
                  <a href="{{ year | prepend: '/blog/' | prepend: site.baseurl}}">
                    <i class="fa-solid fa-calendar fa-sm"></i> {{ year }}
                  </a>
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>
    {% endfor %}
  </div>
</div>
<hr>
{% endif %}

<ul class="post-list">
  {% if page.pagination.enabled %}
    {% assign postlist = paginator.posts %}
  {% else %}
    {% assign postlist = site.posts %}
  {% endif %}

  {% assign postlist = postlist | sort: 'date' | reverse | sort: 'importance' %}
  {% for post in postlist %}

    {% if post.external_source == blank %}
      {% assign text_body = post.content | split: '<div id="references-section">' | first %}
      {% assign words = text_body | number_of_words %}
      {% assign read_time = words | divided_by: 180 | plus: 1 %}
    {% else %}
      {% assign read_time = post.feed_content | strip_html | number_of_words | divided_by: 180 | plus: 1 %}
    {% endif %}
    {% assign year = post.date | date: "%Y" %}
    {% assign tags = post.tags | join: "" %}
    {% assign categories = post.categories | join: "" %}

    <li>
      {% if post.thumbnail %}
        <div class="row align-items-center">
          <div class="col-sm-9">
      {% endif %}

      <h3>
        {% if post.redirect == blank %}
          <a class="post-title" href="{{ post.url | relative_url }}">{{ post.title }}</a>
        {% elsif post.redirect contains '://' %}
          <a class="post-title" href="{{ post.redirect }}" target="_blank" rel="noopener noreferrer">{{ post.title }}</a>
          <svg width="2rem" height="2rem" viewBox="0 0 40 40" xmlns="http://www.w3.org/2000/svg">
            <path d="M17 13.5v6H5v-12h6m3-3h6v6m0-6-9 9" class="icon_svg-stroke" stroke="#999" stroke-width="1.5" fill="none" fill-rule="evenodd" stroke-linecap="round" stroke-linejoin="round"></path>
          </svg>
        {% else %}
          <a class="post-title" href="{{ post.redirect | relative_url }}">{{ post.title }}</a>
        {% endif %}
      </h3>

      <p>{{ post.description }}</p>

      {% if post.community_article_url or post.blogpost_url or post.thesis_url or post.paper_url or post.code_url %}
        <p class="post-links">
          {% if post.community_article_url %}
            <a href="{{ post.community_article_url }}" title="Community Article" target="_blank" rel="noopener noreferrer">
              <i class="fa-brands fa-hugging-face"></i>
            </a>
          {% endif %}
          {% if post.blogpost_url %}
            <a href="{{ post.blogpost_url }}" title="Blogpost" target="_blank" rel="noopener noreferrer">
              <i class="fa-regular fa-newspaper"></i>
            </a>
          {% endif %}
          {% if post.thesis_url %}
            <a href="{{ post.thesis_url }}" title="Thesis" target="_blank" rel="noopener noreferrer">
              <i class="fa-solid fa-graduation-cap"></i>
            </a>
          {% endif %}
          {% if post.paper_url %}
            <a href="{{ post.paper_url }}" title="Paper" target="_blank" rel="noopener noreferrer">
              <i class="fa-solid fa-file-lines"></i>
            </a>
          {% endif %}
          {% if post.code_url %}
            <a href="{{ post.code_url }}" title="Code" target="_blank" rel="noopener noreferrer">
              <i class="fa-brands fa-github"></i>
            </a>
          {% endif %}
        </p>
      {% endif %}

      <p class="post-meta">
        {{ read_time }} min read &nbsp; &middot; &nbsp;
        {{ post.date | date: '%B %d, %Y' }}
        {% if post.external_source %}
          &nbsp; &middot; &nbsp; {{ post.external_source }}
        {% endif %}
      </p>

      <p class="post-tags">
        <a href="{{ year | prepend: '/blog/' | prepend: site.baseurl}}">
          <i class="fa-solid fa-calendar fa-sm"></i> {{ year }}
        </a>

        {% if tags != "" %}
          &nbsp; &middot; &nbsp;
          {% for tag in post.tags %}
            <a href="{{ tag | slugify | prepend: '/blog/tag/' | prepend: site.baseurl}}">
              <i class="fa-solid fa-hashtag fa-sm"></i> {{ tag }}
            </a>
            {% unless forloop.last %}&nbsp;{% endunless %}
          {% endfor %}
        {% endif %}

        {% if categories != "" %}
          &nbsp; &middot; &nbsp;
          {% for category in post.categories %}
            <a href="{{ category | slugify | prepend: '/blog/category/' | prepend: site.baseurl}}">
              <i class="fa-solid fa-tag fa-sm"></i> {{ category }}
            </a>
            {% unless forloop.last %}&nbsp;{% endunless %}
          {% endfor %}
        {% endif %}
      </p>

      {% if post.thumbnail %}
          </div>
          <div class="col-sm-3 post-thumb-col">
            <img class="card-img post-list-thumb" src="{{ post.thumbnail | relative_url }}" alt="image">
          </div>
        </div>
      {% endif %}
    </li>
  {% endfor %}
</ul>

{% if page.pagination.enabled %}
  {% include pagination.liquid %}
{% endif %}

</div>
